for f in *.m4a
do
	ffmpeg -i $f -ar 16000 -ac 1 -acodec pcm_s16le $f.wav
done
