## What to run

    nohup bash collab_stage2_run.sh > /dev/null 2>&1 &
    tail -f stage2_console.log

Total ~3 hours, unattended.

## What to send back

    collab_stage2_results.tar.gz

## If anything goes wrong

The `collab_stage2_results.tar.gz` file is written even when a failure occurs. So please also send it if that is the case. Also, please send `stage2_console.log`.
