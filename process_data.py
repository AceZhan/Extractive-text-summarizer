import math
import time
from os import listdir
from re import split
from nltk import tokenize
from rouge_score import rouge_scorer
import itertools

# 3 minutes to test on the multi_tests
# 16 seconds to test on the single_test

MAX_COMB_L = 5
MAX_COMB_NUM = 100000

class Document(object):
   def __init__(self, story_sents, highlight_sents):
      self.story_sents = story_sents
      self.highlight_sents = highlight_sents
      self.story_len = len(self.story_sents)
      self.highlight_len = len(self.highlight_sents)

def c_n_x(n, x):
   if x > (n >> 2):
      x = n - x
   res = 1
   for i in range(n, n - x, -1):
      res *= i
   for i in range(x, 0, -1):
      res = res // i
   return res

def load_doc(filename):
   file = open(filename)
   text = file.read()
   file.close()
   return text

def split_story(doc):
   index = doc.find('@highlight')
   storyLines, highlights = doc[:index].split('\n'), doc[index:].split('@highlight')
   highlights = [h.strip() for h in highlights if len(h) > 0]
   story = [s.strip() for s in storyLines if len(s) > 0]
   storySentences = []
   for i in range(0, len(story)):
      storySentences += tokenize.sent_tokenize(story[i])

   return Document(storySentences, highlights)

def convert_list_to_sentence(arr):
   return " . ".join(arr) + ' .'

def get_extractive_summary(doc):
   if doc.story_len == 0 or doc.highlight_len == 0:
      return None, 0

   golden_standard = convert_list_to_sentence(doc.highlight_sents)

   sentence_bigram_recall = [0] *  doc.story_len
   scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
   for index, sent in enumerate(doc.story_sents):
      # compare each sentence against golden standard
      # scorer scores with score(target, prediction)
      scores = scorer.score(golden_standard, sent)
      # print(scores)
      recall = scores['rouge2'].recall
      sentence_bigram_recall[index] = recall

   candidates = []
   for index, recall in enumerate(sentence_bigram_recall):
      if recall > 0:
         candidates.append(index)

   all_best_l = 0
   all_best_score = 0
   all_best_comb = None
   for l in range (1, len(candidates)):
      if l > MAX_COMB_L:
         print('Exceed MAX_COMB_L')
         break
      comb_num = c_n_x(len(candidates), l)
      if math.isnan(comb_num) or math.isinf(comb_num) or comb_num > MAX_COMB_NUM:
         print('Exceed MAX_COMB_NUM')
         break
      combs = itertools.combinations(candidates, l)
      l_best_score = 0
      l_best_choice = None
      for comb in combs:
         comb_string = convert_list_to_sentence([doc.story_sents[index] for index in comb])
         rouge_scores = scorer.score(golden_standard, comb_string)
         rouge_bigram_f1 = rouge_scores['rouge2'].fmeasure
         if rouge_bigram_f1 > l_best_score:
            l_best_score = rouge_bigram_f1
            l_best_choice = comb
      if l_best_score > all_best_score:
         all_best_l = l
         all_best_score = l_best_score
         all_best_comb = l_best_choice
      else:
         if l > all_best_l:
            break

   return all_best_comb, all_best_score

def load_stories(directory):
   stories = []
   for name in listdir(directory):
      filename = directory + '/' + name
      doc = load_doc(filename)
      document = split_story(doc)
      comb = get_extractive_summary(document)

      print(document.highlight_sents)
      print([document.story_sents[idx] for idx in comb[0]])

      # stories.append({'story': story, 'hightlights': highlights})
   return stories


start_time = time.time()
stories = load_stories("./cnn_stories_tokenized/multi_test")
print("--- %s seconds ---" % (time.time() - start_time))
