from os import listdir
import time

def load_doc(filename):
    file = open(filename)
    text = file.read()
    file.close()
    return text

def get_story(doc):
    index = doc.find('@highlight')
    storyLines = doc[:index].split('\n')
    return storyLines

def load_stories(directory):
    stories = []
    f = open("corpus.txt", "w")
    for name in listdir(directory):
        filename = directory + '/' + name
        doc = load_doc(filename)
        story = get_story(doc)
        for i in range (len(story)):
            if i == 0:
                f.write(story[i].strip())
            else:
                f.write(' ' + story[i].strip())
        f.write('\n')
    f.close()

start_time = time.time()
load_stories("../cnn_stories_tokenized/full_data_set/")
# load_stories("./cnn_stories_tokenized/single_test")
# load_stories("./cnn_stories_tokenized/multi_test")
print("--- %s seconds ---" % (time.time() - start_time))

