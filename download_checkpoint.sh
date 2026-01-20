#!/bin/bash

echo "----------------------- Downloading pretrained model -----------------------"

model=$1

if [[ $model == "kitti-360" ]]
then
  cp_link="https://cvg.cit.tum.de/webshare/g/behindthescenes/kdbts/kitti_360.zip"
  cp_download_path="out/kitti_360/kitti_360.zip"
  unzip $cp_download_path -d out/kitti_360
elif [[ $model == "kitti-raw" ]]
then
  cp_link="https://cvg.cit.tum.de/webshare/g/behindthescenes/kdbts/kitti_raw.zip"
  cp_download_path="out/kitti_raw/kitti_raw.zip"
  unzip $cp_download_path -d out/kitti_raw
else
  echo Unknown model: $model
  echo Possible options: \"kitti-360\", \"kitti-raw\"
  exit
fi

basedir=$(dirname $0)
outdir=$(dirname $cp_download_path)

cd $basedir || exit
echo Operating in \"$(pwd)\".
echo Creating directories.
mkdir -p $outdir
echo Downloading checkpoint from \"$cp_link\" to \"$cp_download_path\".
wget -O $cp_download_path $cp_link