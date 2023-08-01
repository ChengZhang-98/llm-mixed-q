#!/usr/bin/bash

if [ -z $1 ]; then
    echo "❗ Use timestamp as the backup tag"
    tag=$(date +%Y-%m-%d_%H-%M-%S)
else
    tag=$1
fi

src_dir=$HOME/Projects/llm-mixed-q/checkpoints
bk_dir=$HOME/Projects/llm-mixed-q/local_backup && mkdir -p $bk_dir
bk_zip=$bk_dir/checkpoints_$tag.zip

echo "📂 Backup checkpoints to $bk_zip"
zip -r -1 $bk_zip $src_dir -x *.bin
echo "✅ Done"
