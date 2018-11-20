import mxnet as mx
from mxnet import ndarray as nd
import numpy as np
from collections import namedtuple
import math
Batch = namedtuple('Batch', ['data'])

ctx = mx.gpu(0)
use_batch=False
num_runs=1000


sym, arg_params, aux_params = mx.model.load_checkpoint('/incubator-mxnet/scala-package/examples/scripts/infer/models/resnet-152/resnet-152', 0)
mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
if use_batch:
    mod.bind(for_training=False, data_shapes=[('data', (16,3,224,224))],
             label_shapes=mod._label_shapes)
else:
    mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))],
             label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)

with open('/incubator-mxnet/scala-package/examples/scripts/infer/models/resnet-152/synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]



def pre_process_image(path):
        img = mx.image.imread(path)
        if img is None:
                return None

        img = mx.image.imresize(img, 224, 224) # resize
        img = img.transpose((2, 0, 1)) # Channel first
        img = img.expand_dims(axis=0) # batchify
        #img = nd.random.poisson(1.0, shape=(1, 3, 512, 512))
        a = nd.concat(img, dim = 0)
        if use_batch:
                for i in range(1, 16):
                        a = nd.concat(a, img, dim = 0)
        return a.as_in_context(ctx)

def predict(img):
    # compute the predict probabilities
    import time
    # print img.shape
    data_iter = None
    if use_batch:
        data_iter = mx.io.NDArrayIter([img], None, 16)
    else:
        data_iter = mx.io.NDArrayIter([img], None, 1)

    start = time.time()

    op = mod.predict(data_iter)
    #mod.forward(Batch([img]))
    op.wait_to_read()
#    print (type(op[0]))
    end = time.time()

 #   prob = op[0]
#    print (op[0])
    #op[0].copyto(prob)

    #prob = prob.asnumpy()
    # print (mod.get_outputs()[0].shape)
    # print len(mod.get_outputs())
    #prob = mod.get_outputs()[0]
    #shape = mod.get_outputs()[0].shape
    print (end - start)
    
    #prob = np.squeeze(prob)
    #a = np.argsort(prob)[::-1]
    #for i in a[0:5]:
     #   print('probability=%f, class=%s' %(prob[i], labels[i]))
    # print the top-5
#    prob = np.squeeze(prob)
 #   a = np.argsort(prob)[::-1]
    # print (len(prob))
    return end - start


def percentile(val, arr):
        idx = int(math.ceil((len(arr) - 1) * val / 100.0))
        return arr[idx]

times = list()
img = pre_process_image('/incubator-mxnet/scala-package/examples/scripts/infer/images/dog.jpg')
for i in range(1, num_runs):
        times.append(predict(img))

times.sort()
# print times
p99 = percentile(99, times) * 1000
p90 = percentile(90, times) * 1000
p50 = percentile(50, times) * 1000
average = sum(times)/len(times) * 1000
print ("p99 in ms, %f" % p99)
print ("p90 in ms, %f" % p90)
print ("p50 in ms, %f" % p50)
print ("average in ms, %f" % average)