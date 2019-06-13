import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_integer("topK", None,
                    "The topK item that the reco model was trained to recommend.")
tf.app.flags.DEFINE_integer('latentFactorNum',20,'number of latent Factors to use')

class Tester():
    def __init__(self,file1=1,file2=2,latentFactorNum=3):
        self.file1=file1
        self.file2=file2
        self.latentFactorNum=latentFactorNum
    def display(self):
        print('latentFactorNum is: ',self.latentFactorNum)
def main(_):
    tester = Tester(latentFactorNum=FLAGS.latentFactorNum)
    tester.display()
if __name__=="__main__":
    tf.app.run()
