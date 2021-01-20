from transformers import pipeline
import numpy as np
from nltk import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer

nlp = pipeline('sentiment-analysis')


def preprocess(s):
	s = s.replace("\n", '')
	s = s.replace("\\n", '')
	s = s.replace("\t", '')
	s = s.replace("\\", '')
	return sent_tokenize(s)


def sentiment_(x):
	scores = {'POSITIVE': [],
				'NEGATIVE': []}
	for sent in x:
		pred = nlp(sent)[0]
		scores[pred['label']].append(pred['score'])
	return (np.sum(scores['POSITIVE']) - np.sum(scores['NEGATIVE']))/len(x)


review_txt = "In this paper the authors offer a new algorithm to detect cancer \
mutations from sequencing cell free DNA (cfDNA). The idea is that in the sample \
being sequenced there would also be circulating tumor DNA (ctDNA) so such mutations \
could be captured in the sequencing reads. The issue is that the ctDNA are expected \
to be found with low abundance in such samples, and therefore are likely to be hit \
by few or even single reads. This makes the task of differentiating between sequencing \
errors and true variants due to ctDNA hard. The authors suggest to overcome this problem \
by training an algorithm that will identify the sequence context that characterize sequencing \
errors from true mutations. To this, they add channels based on low base quality, low mapping \
quality. The algorithm for learning the context of sequencing reads compared to true mutations is \
based on a multi layered CNN, with 2/3bp long filters to capture di and trinucleotide frequencies, \
and a fully connected layer to a softmax function at the top. The data is based on mutations in 4 \
patients with lung cancer for which they have a sample both directly from the tumor and from a \
healthy region. One more sample is used for testing and an additional cancer control which is not \
lung cancer is also used to evaluate performance.Pros:\
The paper tackles what seems to be both an important and challenging problem. \
We also liked the thoughtful construction of the network and way the reference, \
the read, the CIGAR and the base quality were all combined as multi channels to \
make the network learn the discriminative features of from the context. Using matched \
samples of tumor and normal from the patients is also a nice idea to mimic cfDNA data. \
Cons: \
While we liked both the challenge posed and the idea to solve it we found several major issues with the work. \
First, the writing is far from clear. There are typos and errors all over at an unacceptable level. Many terms are not defined or defined after being introduced (e.g. CIGAR, MF, BQMQ). A more reasonable CS style of organization is to first introduce the methods/model and then the results, but somehow the authors flipped it and started with results first, lacking many definitions and experimental setup to make sense of those.  Yet Sec. 2 “Results” p. 3 is not really results but part of the methods. The “pipeline” is never well defined, only implicitly in p.7 top, and then it is hard to relate the various figures/tables to bottom line results (having the labels wrong does not help that).\
The filters by themselves seem trivial and as such do not offer much novelty. Moreover, the authors filter the “normal” samples using those (p.7 top), which makes the entire exercise a possible circular argument. \
If the entire point is to classify mutations versus errors it would make sense to combine their read based calls from multiple reads per mutations (if more than a single read for that mutation is available) - but the authors do not discuss/try that. \
The entire dataset is based on 4 patients. It is not clear what is the source of the other cancer control case. The authors claim the reduced performance show they are learning lung cancer-specific context. What evidence do they have for that? Can they show a context they learned and make sense of it? How does this relate to the original papers they cite to motivate this direction (Alexandrov 2013)? Since we know nothing about all these samples it may very well be that that are learning technical artifacts related to their specific batch of 4 patients. As such, this may have very little relevance for the actual problem of cfDNA. \
Finally, performance itself did not seem to improve significantly compared to previous methods/simple filters, and the novelty in terms of ML and insights about learning representations seemed limited.\
Albeit the above caveats, we iterate the paper offers a nice construction for an important problem. We believe the method and paper could potentially be improved and make a good fit for a future bioinformatics focused meeting such as ISMB/RECOMB."

#"The paper is relatively clear to follow, and implement. The main concern is that this looks like a class project rather than a scientific paper. For a class project this could get an A in a ML class!In particular, the authors take an already existing dataset, design a trivial convolutional neural network, and report results on it. There is absolutely nothing of interest to ICLR except for the fact that now we know that a trivial network is capable of obtaining 90\% accuracy on this dataset."

print(sentiment_(preprocess(review_txt)))

