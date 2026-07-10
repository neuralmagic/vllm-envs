#!/bin/bash
# Release-history sweep acceptance test for ve (see plan: verification #9).
# Usage: release_sweep.sh <ref>... ; results in /tmp/ve-sweep/results.tsv
set -u
export VE_MAX_SIZE_GB=200
LOGDIR=/tmp/ve-sweep
RESULTS=$LOGDIR/results.tsv
MODEL=facebook/opt-125m
PORT=8199
mkdir -p "$LOGDIR"
printf 'pass\tref\tve_new_s\tattach\tserve\tcompletion\n' > "$RESULTS"

attach_mode() {  # classify how the build layer resolved from the ve log
    if grep -q "cache HIT" "$1" && ! grep -q "build layer: cache MISS" "$1"; then
        echo store-hit
    elif grep -q "fetching precompiled wheel" "$1"; then
        echo precompiled
    elif grep -q "building extensions" "$1"; then
        echo local-build
    else
        echo unknown
    fi
}

cleanup_gpu() {
    pkill -9 -f "api_server" 2>/dev/null || true
    pkill -9 -f "vllm serve" 2>/dev/null || true
    pkill -9 -f "VLLM::" 2>/dev/null || true
    sleep 5
}

serve_check() {  # $1=env dir  $2=log; echoes "serve completion"
    local env=$1 log=$2 serve=fail comp=fail
    # run from inside the env: vLLM's registry inspection spawns `python -m
    # vllm...`, and python puts cwd first on sys.path — launching from another
    # vLLM checkout imports that checkout's source into this env's venv
    (cd "$env" && CUDA_VISIBLE_DEVICES=${SWEEP_GPU:-7} timeout 45m \
        .venv/bin/vllm serve "$MODEL" --port $PORT) > "$log" 2>&1 &
    local pid=$!
    for _ in $(seq 1 180); do
        if curl -sf "localhost:$PORT/v1/models" > /dev/null 2>&1; then
            serve=ok; break
        fi
        kill -0 $pid 2>/dev/null || break
        sleep 5
    done
    if [ $serve = ok ]; then
        local resp
        resp=$(curl -sf "localhost:$PORT/v1/completions" \
            -H 'Content-Type: application/json' \
            -d '{"model":"'"$MODEL"'","prompt":"Hello, my name is","max_tokens":8}')
        echo "$resp" | grep -q '"text"' && comp=ok
    fi
    kill $pid 2>/dev/null
    cleanup_gpu
    echo "$serve $comp"
}

run_ref() {  # $1=pass  $2=ref  $3=do_serve
    local pass=$1 ref=$2 do_serve=$3
    local name="sweep-$(echo "$ref" | sed 's/[^A-Za-z0-9._-]/-/g' | cut -c1-24)"
    local velog="$LOGDIR/$name.pass$pass.ve.log"
    local t0=$SECONDS
    if ! ve new "$ref" --name "$name" --repo /home/LucasWilkinson/local/vllm2 \
            > "$velog" 2>&1; then
        printf '%s\t%s\t%d\tve-new-FAIL\t-\t-\n' "$pass" "$ref" $((SECONDS - t0)) >> "$RESULTS"
        ve rm "$name" > /dev/null 2>&1
        return
    fi
    local dt=$((SECONDS - t0))
    local mode; mode=$(attach_mode "$velog")
    local serve=- comp=-
    if [ "$do_serve" = yes ]; then
        read -r serve comp <<< "$(serve_check "$HOME/vllm-envs/$name" "$LOGDIR/$name.serve.log")"
    fi
    printf '%s\t%s\t%d\t%s\t%s\t%s\n' "$pass" "$ref" "$dt" "$mode" "$serve" "$comp" >> "$RESULTS"
    ve rm "$name" > /dev/null 2>&1
}

echo "[sweep] pass 1 (cold): $*"
for ref in "$@"; do
    echo "[sweep] === pass1 $ref ==="
    run_ref 1 "$ref" yes
    column -t "$RESULTS" | tail -1
done

echo "[sweep] pass 2 (expect all cache hits, no serve)"
for ref in "$@"; do
    echo "[sweep] === pass2 $ref ==="
    run_ref 2 "$ref" no
    column -t "$RESULTS" | tail -1
done

echo "[sweep] done"
column -t "$RESULTS"
ve du
