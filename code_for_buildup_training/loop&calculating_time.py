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
