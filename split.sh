#! /bin/bash

LENGTH=2 # seconds

cd ./songs
rm -vfR split
mkdir -v split
for file in *.mp3;
do
	base_filename=$(basename "$file" .mp3);
	printf -v new_filename '%s' "${base_filename//[^a-zA-Z0-9_-]}"
	ffmpeg -i "$file" -f segment -segment_time ${LENGTH} -c copy "split/${new_filename}_%05d.wav";
done
cd ..

cd ./speech
rm -vfR split
mkdir -v split
for file in *.mp3;
do
	base_filename=$(basename "$file" .mp3);
	printf -v new_filename '%s' "${base_filename//[^a-zA-Z0-9_-]}"
	ffmpeg -i "$file" -f segment -segment_time ${LENGTH} -c copy "split/${new_filename}_%05d.wav";
done
