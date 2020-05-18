from scipy.io import wavfile
from os import listdir
from IPython.display import display, Markdown, Latex
from scipy.signal import spectrogram, lfilter, freqz, tf2zpk, get_window
from scipy.stats import pearsonr
from numpy import transpose, log10, arange
import matplotlib.pyplot as plt



sentencesFilePath = "../sentences/"
queriesFilePath = "../queries/"

class AudioFile:
	def __init__(self, path, fileName):
		fs, data = wavfile.read(path + fileName)
		self.name = fileName[:-4]
		if fileName == "q1.wav":
			self.name = "disapproves"
		if fileName == "q2.wav":
			self.name = "occasionally"
		self.fs = fs
		self.data = data
		self.spectrogram = self.generateSpectrogram()
		self.features = self.generateFeatures()

	def generateSpectrogram(self):
		wlen = int(25e-3*self.fs)
		wshift = int(10e-3*self.fs)
		woverlap = wlen - wshift
		window = get_window("hamming", wlen)
		f, t, sgr = spectrogram(self.data, self.fs, window, wlen, woverlap, 512)
		sgr_log = 10 * log10(sgr+1e-20)
		return f, t, sgr_log

	def generateFeatures(self):
		fsgr, tsgr, sgr = self.spectrogram
		final = []
		tmp = [0 for i in sgr[0]]
		f = [f for i, f in enumerate(fsgr) if i%16 == 0]
		f.pop(-1)
		for i, spec in enumerate(sgr):
			if i%16 == 0 and i != 0:
				final.append(tmp)
				tmp = spec
			else:
				for j, value in enumerate(spec):
					tmp[j] += value
		return f, tsgr, final

	def scoreFor(self, query, sampleNumber):
		q = transpose(query.features[2])
		f = transpose(self.features[2])
		score = []
		if 0 > sampleNumber or sampleNumber > len(f) - len(q):
			return None
		delta = int(sampleNumber)
		fc = f[delta: delta + len(q)]
		p = 0.0
		for i in range(0, len(q)):
			p += pearsonr(q[i], fc[i])[0]
		return p / len(q)



files = listdir(sentencesFilePath)
sentences = [AudioFile(sentencesFilePath, file) for file in listdir(sentencesFilePath)]
queries = [AudioFile(queriesFilePath, file) for file in listdir(queriesFilePath)]


def getMarkDownTableRowFor(array, width):
	strs = [str(e) for e in array]
	formated = [width*" " for i in range(0, len(array))]
	for j in range(0, len(array)):
		string = strs[j]
		for i in range(0, width):
			if i >= len(string):
				break
			tmp = formated[j]
			tmp = tmp[:i] + string[i] + tmp[i+1:]
			formated[j] = tmp
	return "|" + "|".join(formated) + "|\n"

def table(audioFiles, function, headers):
	width = 128
	mdText = getMarkDownTableRowFor(headers, width)
	mdText += getMarkDownTableRowFor([width*"-" for d in headers], width)
	data = [function(audio) for audio in audioFiles]
	mdText += "".join([getMarkDownTableRowFor(d, width) for d in data])
	display(Markdown(mdText))

def plotSpectrogram(spectrogram, name):
	f, t, sgr_log = spectrogram
	plt.figure(figsize=(9,3))
	plt.pcolormesh(t,f,sgr_log)
	plt.title(name)
	plt.gca().set_xlabel('Čas [s]')
	plt.gca().set_ylabel('Frekvence [Hz]')
	plt.tight_layout()

def plotStatistics(audio):
    plt.figure(figsize=(9,9))
    queryNames = [q.name for q in queries]
    name = "\"" + "\" and \"".join(queryNames) + "\" vs. " + audio.name
    plt.subplot(311)
    plt.title(name)
    plt.gca().set_ylabel('signal')
    plt.gca().set_xlabel('t')
    time = float(len(audio.data)/audio.fs)
    t1 = arange(0.0, time, 1/audio.fs)
    plt.plot(t1, audio.data)
    
    f, t, sgr_log = audio.features
    plt.subplot(312)
    x = [i for i in range(16)]
    plt.pcolormesh(t,x,sgr_log)
    plt.gca().set_ylabel('features')
    plt.gca().set_xlabel('t')
    plt.tight_layout()
    
    plt.subplot(313)
    plt.gca().set_ylabel('signal')
    plt.gca().set_xlabel('t')
    time = float(len(audio.data)/audio.fs)
    queryDistribution = []
    t2 = arange(0.0, len(audio.data), 0.05 * audio.fs)
    for query in queries:
        queryLength = len(query.features[2][0])
        y = []
        x = []
        for i, time in enumerate(audio.features[1]):
            if i%5 == 0:
                score = audio.scoreFor(query, i)
                if score != None:
                	x.append(time)
                	y.append(score)
        function, = plt.plot(x, y, label=query.name)
        queryDistribution.append(function)
    plt.legend(queryDistribution, loc='upper right')
    plt.show()

