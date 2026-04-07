$Remote = "<USER>@monsoon.hpc.nau.edu"
$RemoteBase = "/scratch/<USER>/evals/cocci_eval/seeded_runs"
$LocalBase = "localdata/evals/cocci_eval/seeded_runs"

$Models = @("flagship1", "flagship2")
$Sets = 1..10 | ForEach-Object { "set_{0:D2}" -f $_ }
$Artifacts = @("evaluation", "prediction", "peptide_compare")

$Copied = 0
$Failed = 0

foreach ($Model in $Models) {
    foreach ($Set in $Sets) {
        $LocalCombined = Join-Path $LocalBase "$Model/$Set/combined"
        New-Item -ItemType Directory -Force -Path $LocalCombined | Out-Null

        foreach ($Artifact in $Artifacts) {
            $RemotePath = "${Remote}:$RemoteBase/$Model/$Set/combined/$Artifact"
            Write-Host "[scp] $RemotePath -> $LocalCombined"

            & scp -r $RemotePath $LocalCombined
            if ($LASTEXITCODE -eq 0) {
                $Copied++
            }
            else {
                Write-Warning "[scp failed] $RemotePath"
                $Failed++
            }
        }
    }
}

Write-Host "[done] copied=$Copied failed=$Failed"
if ($Failed -gt 0) {
    exit 1
}
