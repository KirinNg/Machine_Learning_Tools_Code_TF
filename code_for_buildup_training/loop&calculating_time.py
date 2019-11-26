# cal time
import datetime

for e in range(5):
    begin_time = datetime.datetime.now()
    end_time = datetime.datetime.now()
    during_time = end_time - begin_time
    print(during_time.microseconds)


# trick for tqgm
import tqdm
pbar = tqdm.trange(iter)
for k in pbar:
    pbar.set_description("hello!")


# pickle
import pickle
def save_batch(self, path, batch=None, batchsize=50):
    if not batch:
        batch = self.get_batch(batchsize)
    file_batch = open(path, 'wb')
    pickle.dump(batch, file_batch)
    file_batch.close()

def load_batch(self, path):
    try:
        file_batch = open(path, 'rb')
        batch = pickle.load(file_batch)
        file_batch.close()
        return batch
    except:
        print("you haven't saved the batch")