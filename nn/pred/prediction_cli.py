"""prediction_cli.py

This module is designed to predict the probability of a peptide belonging to three different classes: definitely epitope, uncertain, and not epitope by using a PepSeqFFNN.

***Usage TBD
"""
import argparse
from pathlib import Path
import torch
import esm
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Iterator, List
from pipelineio.logger import setup_logger
from esm2.embeddings import clean_seq, compute_window_embedding, append_seq_len
from nn.models.ffnn import PepSeqFFNN

CLASS_NAMES = ["Def epitope", "Uncertain", "Not epitope"]

def find_peptide_start_stop(protein_seq: str, peptide: str) -> Tuple[int, int]:
    """
    Find the peptide start and stop indices within an overall protein sequence if they were not provided.

    Parameters
    ----------
        protein_seq : str
            The entire protein sequence used to generate the peptide.
        peptide : str
            The peptide as a string which should be a subset of the overall protein sequence.

    Returns
    -------
        start : int
            Start index of the peptide.
        stop : int
            Stop index of the peptide.

    Raises
    ------
        ValueError
            If peptide is not found in the protein sequence.
    """
    start = protein_seq.find(peptide)
    stop = len(peptide) + start
    if start < 0 or stop > len(protein_seq):
        raise ValueError("Peptide not found as a contiguous substring in the provided protein sequence")
    return start, stop

def infer_emb_dim(state: Dict[str, Any]) -> int:
    """
    Infers the embedding dimension from the model state.

    Parameters
    ----------
        state : Dict[str, Any]
            The current model state as a dict.

    Returns
    -------
        int
            The embedding dimension (usually 1281).

    Raises
    ------
        ValueError
            If no 2D weight tensors were found to infer embedding dimension.
    """
    for _, v in state.items():
        if torch.is_tensor(v) and v.dim() == 2:
            return int(v.shape[1])
    raise ValueError("Could not infer embedding dimension from model checkpoint, no 2D weight tensors found")

def read_fasta_pep_prot(fasta_path: Path | str) -> Iterator[Tuple[str, str]]:
    peptide = None
    seq_lines = []
    with open(fasta_path ,"r", encoding="utf-8") as fasta:
        for raw in fasta:
            line = raw.strip()

            if not line:
                continue

            if line.startswith(">"):
                if peptide:
                    protein_seq = clean_seq("".join(seq_lines))
                    pep = clean_seq(peptide)
                    yield pep, protein_seq

                peptide = line[1:].strip()
                seq_lines = []
            
            else:
                seq_lines.append(line)

        if peptide:
            protein_seq = clean_seq("".join(seq_lines))
            pep = clean_seq(peptide)
            yield pep, protein_seq
    
def embed_protein_seq(protein_seq: str, 
                      model_name: str = "esm2_t33_650M_UR50D", 
                      max_tokens: int = 1022) -> torch.Tensor:
    """
    Generates the embedding for an entire protein sequence.

    Parameters
    ----------
        protein_seq : str
            The overall protein sequence the peptide was generated from.
        model_name : str
            The model used to generate ESM-2 training embeddings from. This 
            model must match the training model. Default is esm2_t33_650M_UR50D.
        max_tokens : int
            Maximum number of tokens the ESM model can fit in its context window. Default is 1022.

    Returns
    -------
        Tensor
            The entire protein sequence's embeddings.
    """
    # load ESM embedding model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, alphabet = getattr(esm.pretrained, model_name)()
    model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    # get batched tokens using one sequence passed
    pairs = [("query", protein_seq)]
    _, _, batch_tokens = batch_converter(pairs)

    # set model layer to use (33 for default model)
    layer = 33 if "t33" in model_name else 30 if "t30" in model_name else 12 if "t12" in model_name else 6

    seq_len = len(protein_seq)
    token_len = batch_tokens.size(1)

    with torch.inference_mode():
        if token_len <= (max_tokens + 2):
            if device.startswith("cuda") and torch.cuda.is_available():
                batch_tokens = batch_tokens.to(device, non_blocking=True)
            out = model(batch_tokens, repr_layers=[layer], return_contacts=False)
            rep = out["representations"][layer][0, 1:1 + seq_len].to("cpu")
        
        else:
            rep = compute_window_embedding(batch_tokens, model, layer, device)

    rep_np = rep.numpy().astype(np.float32)
    rep_np = append_seq_len(rep_np, seq_len)

    return torch.from_numpy(rep_np)

def predict(model: PepSeqFFNN, 
            protein_seq: str, 
            peptide: str, 
            model_name: str, 
            max_tokens: int, 
            device: str) -> Dict[str, Any]:
    protein_seq = clean_seq(protein_seq)
    peptide = clean_seq(peptide)

    align_start, align_stop = find_peptide_start_stop(protein_seq, peptide)
        
    # generate embedding and get peptide-level embedding
    protein_emb = embed_protein_seq(protein_seq, model_name=model_name, max_tokens=max_tokens)
    peptide_emb = protein_emb[align_start:align_stop, :]
    X = peptide_emb.unsqueeze(0).to(device, non_blocking=True)

    # generate probability outputs (predictions)
    with torch.inference_mode():
        logits = model(X) # (1, 3)
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()

    pred_idx = int(torch.tensor(probs).argmax().item())

    return {"peptide": peptide, 
            "protein": protein_seq, 
            "p_def_epitope": probs[0], 
            "p_uncertain": probs[1], 
            "p_not_epitope": probs[2], 
            "pred_class": CLASS_NAMES[pred_idx], 
            "pred_index": pred_idx, 
            "max_prob": max(probs)}

def main() -> None:
    """Handles command-line argument parsing and high-level execution of the Prediction program."""
    parser = argparse.ArgumentParser(description="Predict epitope class for a peptide using a trained PepSeqPred model.")
    parser.add_argument("checkpoint", 
                        type=Path, 
                        help="Path to the trained model checkpoint (.pt).")
    parser.add_argument("fasta_input", 
                        type=Path, 
                        help="FASTA file where header is a peptide and body is a protein sequence.")
    parser.add_argument("--output-csv", 
                        action="store", 
                        dest="output_csv", 
                        type=Path, 
                        default=Path("predictions.csv"), 
                        help="Output CSV path for predictions"
                        )
    # parser.add_argument("protein_seq",  
    #                     type=str, 
    #                     help="Full protein sequence the peptide came from.")
    # parser.add_argument("peptide", 
    #                     type=str, 
    #                     help="Peptide sequence to score.")
    # parser.add_argument("--align-start", 
    #                     action="store", 
    #                     dest="align_start", 
    #                     type=int, 
    #                     default=None, 
    #                     help="0-based peptide start index in the protein.")
    # parser.add_argument("--align-stop", 
    #                     action="store", 
    #                     dest="align_stop", 
    #                     type=int, 
    #                     default=None, 
    #                     help="0-based peptide stop index in the protein.")    
    parser.add_argument("--model-name", 
                        action="store", 
                        dest="model_name", 
                        type=str, 
                        default="esm2_t33_650M_UR50D", 
                        help="ESM model name to generate embeddings.")
    parser.add_argument("--max-tokens", 
                        action="store", 
                        dest="max_tokens", 
                        type=int, 
                        default=1022, 
                        help="ESM residue token budget excluding CLS and EOS.")
    parser.add_argument("--log-dir", 
                        action="store", 
                        dest="log_dir", 
                        type=Path, 
                        default=Path("logs"), 
                        help="Directory for logs.")
    parser.add_argument("--log-level", 
                        action="store", 
                        dest="log_level", 
                        type=str, 
                        default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                        help="Logging level choice.")
    parser.add_argument("--log-json", 
                        action="store_true", 
                        dest="log_json", 
                        default=False, 
                        help="Emit logs as JSON lines for simple parsing.")
    args = parser.parse_args()

    json_indent = 2 if args.log_json else None
    logger = setup_logger(log_dir=args.log_dir, 
                          log_level=args.log_level, 
                          json_lines=args.log_json, 
                          json_indent=json_indent, 
                          name="prediction_cli")
    
    # protein_seq = clean_seq(args.protein_seq)
    # peptide = clean_seq(args.peptide)

    # # get peptide start and stop indices
    # if args.align_start is None and args.align_stop is None:
    #     align_start, align_stop = find_peptide_start_stop(protein_seq, peptide)

    # else:
    #     align_start, align_stop = args.align_start, args.align_stop

    #     if align_start < 0 or align_stop > len(protein_seq):
    #         raise ValueError("Peptide not found as a contiguous substring in the provided protein sequence")
    
    # load model from disk
    checkpt = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(checkpt, dict) or "model_state_dict" not in checkpt:
        raise ValueError("Model checkpoint format unrecognized, expected dictionary with model_state_dict key")
    state = checkpt["model_state_dict"]

    # build model loaded from disk
    emb_dim = infer_emb_dim(state)
    model = PepSeqFFNN(emb_dim=emb_dim, num_classes=len(CLASS_NAMES))
    model.load_state_dict(state)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    logger.info("prediction_init", 
                extra={"extra": {
                    "checkpoint": str(args.checkpoint), 
                    "emb_dim": emb_dim, 
                    "device": device
                }})

    # logger.info("prediction_start", 
    #             extra={"extra": {
    #                 "protein_len": len(protein_seq),
    #                 "peptide_len": len(peptide),
    #                 "align_start": align_start,
    #                 "align_stop": align_stop,
    #                 "esm_model_name": args.model_name,
    #                 "max_tokens": args.max_tokens,
    #                 "emb_dim": emb_dim,
    #                 "device": device
    #             }})

    # handle fasta file input
    rows: List[Dict[str, Any]] = []
    n = 0
    for pep, prot in read_fasta_pep_prot(args.fasta_input):
        try:
            row = predict(model, prot, pep, 
                          args.model_name, args.max_tokens, device)
            rows.append(row)
            n += 1

            # log prediction progress every 10 peptides
            if n % 10:
                logger.info("prediction_progress", 
                            extra={"extra": {
                                "processed": n
                            }})
                
        except Exception as e:
            logger.error("prediction_failed", 
                         extra={"extra": {
                             "peptide": pep, 
                             "error": str(e)
                         }})
    
    # store as DataFrame for easy CSV file creation
    df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    logger.info("prediction_done", 
                extra={"extra": {
                    "output_csv": str(args.output_csv), 
                    "processed": len(df), 
                    "columns": list(df.columns)
                }})

    # protein_emb = embed_protein_seq(protein_seq, 
    #                                 model_name=args.model_name, 
    #                                 max_tokens=args.max_tokens) # on CPU, (L, D + 1)
    # peptide_emb = protein_emb[align_start:align_stop, :]
    # X = peptide_emb.unsqueeze(0).to(device, non_blocking=True) # (1, L_p, D + 1)

    # # generate probability outputs (predictions)
    # with torch.inference_mode():
    #     logits = model(X) # (1, 3)
    #     probs = torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()

    # pred_idx = int(torch.tensor(probs).argmax().item())

    # output = {"align_start": align_start, 
    #           "align_stop": align_stop, 
    #           "peptide": peptide, 
    #           "classes": CLASS_NAMES, 
    #           "probs": probs, 
    #           "pred_class": CLASS_NAMES[pred_idx], 
    #           "pred_index": pred_idx}

    # logger.info("prediction_done", extra={"extra": output})

if __name__ == "__main__":
    main()
