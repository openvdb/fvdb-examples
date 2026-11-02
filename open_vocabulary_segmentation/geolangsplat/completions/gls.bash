# Bash tab-completion for the `gls` CLI.
#
# Install (one time):
#   echo "source /path/to/geolangsplat/completions/gls.bash" >> ~/.bashrc
#   source ~/.bashrc
#
# Then:
#   gls <TAB>                 -> segment bake catalog check doctor serve show status stop render explore
#   gls segment x.ply foo -r sat<TAB>   -> satellite satellite_dense
#   gls segment ... -O <TAB>  -> mask ply_segmented ply_overlay report
#   .ply / paths complete as filenames.

_gls_complete() {
    local cur prev
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    local subcommands="segment bake catalog check doctor serve show status stop render explore"
    local recipes="auto satellite satellite_dense aerial"
    local outputs="mask ply_segmented ply_overlay report"
    local view_sources="render images globe"
    local cache_dtypes="auto amp fp16 bf16"
    local early_stops="auto on off"

    case "$prev" in
        -r|--recipe)
            COMPREPLY=( $(compgen -W "$recipes" -- "$cur") ); return 0 ;;
        -O|--output)
            COMPREPLY=( $(compgen -W "$outputs" -- "$cur") ); return 0 ;;
        --view-source)
            COMPREPLY=( $(compgen -W "$view_sources" -- "$cur") ); return 0 ;;
        --cache-dtype)
            COMPREPLY=( $(compgen -W "$cache_dtypes" -- "$cur") ); return 0 ;;
        --stream-early-stop)
            COMPREPLY=( $(compgen -W "$early_stops" -- "$cur") ); return 0 ;;
        -o|--out-path|-s|--sfm)
            COMPREPLY=( $(compgen -f -- "$cur") ); return 0 ;;
    esac

    # first token after `gls` -> the subcommand
    if [ "$COMP_CWORD" -eq 1 ]; then
        COMPREPLY=( $(compgen -W "$subcommands" -- "$cur") ); return 0
    fi

    # flags when the user has started typing one
    if [[ "$cur" == -* ]]; then
        COMPREPLY=( $(compgen -W "-r --recipe -O --output -o --out-path -t --select --keep-alive -b --background \
            --view-source -s --sfm --compete --profile -d --device \
            --low-vram --no-low-vram --cache-dtype --stream-early-stop --stream-chunk \
            --view-cap --max-views --vram-budget-gb -u --up --fast-views \
            --inside-out --peak -h --help" -- "$cur") )
        return 0
    fi

    # default: filenames (model .ply, output paths)
    COMPREPLY=( $(compgen -f -- "$cur") )
}

complete -o filenames -F _gls_complete gls
