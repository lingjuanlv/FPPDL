require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides a normalization operator
require 'optim'   -- an optimization package, for online and batch methods
require 'dataset-mnist'
-- require 'dataset-svhn'
os.execute('mkdir ' .. 'save')

function cut(value, range)
   local cvalue = value
   if range > 0 then
      if cvalue < -range then
         cvalue = -range
      elseif cvalue > range then
         cvalue = range
      end
   end
   return cvalue
end

-- options
cmd = torch.CmdLine()

cmd:option('-dataset',           'mnist', 'svhn | mnist')
cmd:option('-dataSizeFrac',      1,       'the fraction of dataset to be used for training')
cmd:option('-model',             'cvn',   'convnet | cvn | linear | mlp | deep')
cmd:option('-method',            'syn',   'seq | fla | asy |syn')
cmd:option('-slevel',            1,      '1 | 5') --1: sharing 10% in each epoch; 5: sharing 10%-100% in each epoch
cmd:option('-IID',            1,      '1 | 0')
cmd:option('-imbalanced',        0,      '0 | 1')
cmd:option('-learningRate',      1e-3,    '')
cmd:option('-learningRateDecay', 1e-7,    '')
cmd:option('-batchSize',         1,       '')
cmd:option('-weightDecay',       0,       '')
cmd:option('-momentum',          0,       '')
cmd:option('-threads',           2,       '')
cmd:option('-netSize',           4,      '')
cmd:option('-shardSizeFrac',     0.01,    'fraction of the training set in each shard')
cmd:option('-uploadFraction',    0.1,     'fraction of parameters to be uploaded after training')
cmd:option('-downloadFraction',  1,       'fraction of parameters to be downloaded before training')
cmd:option('-epochFraction',     1,       '')
cmd:option('-epsilon',           0,       'epsilon for dp. 0: disable it')
cmd:option('-delta',             0,       'delta for dp.')
cmd:option('-range',             0.001,       'cut the gradiants between -range and range. 0: disable it')
cmd:option('-threshold',         0,       'release those whose abs value is greater than threshold. 0: disable it')
cmd:option('-nepochs',           60,     '')
cmd:option('-local_nepochs',     1,     '')
cmd:option('-taskID',            '0',     'the ID associated to the task')
cmd:option('-folder',            'save',  '')
cmd:option('-shardID',            '0',     'the ID associated to the shardfile')
cmd:option('-run',            '0',  '')
cmd:option('-credit_thres',      1,  '0 | 1')
cmd:option('-credit_fade',   1,  '0 | 1')
cmd:option('-update_criteria',   'large',  'large | random')
cmd:option('-pretrain',   0,  '0 | 1')
cmd:option('-pretrain_epochs',   10,  '10 | 5')

opt = cmd:parse(arg or {})

shardfile=paths.concat(opt.folder, 'trainshard.' .. opt.shardID .. '.' .. opt.run) 
print(shardfile)
slevelfile = paths.concat(opt.folder, 'share_level.' .. opt.shardID .. '_tpds'  .. '.'  .. opt.run)
print(slevelfile)
psfile = paths.concat(opt.folder, 'ps.' .. opt.taskID .. '.'  .. opt.run)
pspretrain_file = paths.concat(opt.folder, 'pspretrain.' .. opt.taskID .. '.'  .. opt.run)
print(opt)

-- config torch
torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')

-- create training set and test set
if opt.dataset == 'mnist' then
  nbTrainingPatches = 60000
  nbTestingPatches = 10000
  geometry = {32, 32}
  trainData = mnist.loadTrainSet(nbTrainingPatches, geometry)
  testData = mnist.loadTestSet(nbTestingPatches, geometry)

  local mean = trainData.data:mean()
  local std = trainData.data:std()
  trainData.data:add(-mean):div(std)
  testData.data:add(-mean):div(std)
end
trainSize = math.ceil(opt.dataSizeFrac * trainData.labels:size(1))
testSize  = testData.labels:size(1)
print('testSize: '..testSize)

trainData.size = function() return trainSize end
testData.size = function() return testSize end
local test_shuffle = torch.randperm(testSize)


shardSize={}
trainData.shard = {}
if paths.filep(shardfile) then
   trainData.shard= torch.load(shardfile, 'binary')
   print('load shard indices ' .. shardfile)
   for nid = 1, opt.netSize do
      shardSize[nid] = #trainData.shard[nid]
      shardSize[nid] = shardSize[nid][1]
   end
else
  print('create shard indices')
  if opt.imbalanced==0 and opt.IID==1 then
    for nid = 1, opt.netSize do
      shardSize[nid] = math.ceil(opt.shardSizeFrac * trainSize)
      print('balanced shardSize for party '.. nid .. ': ' .. shardSize[nid] ..'\n')
      local shffl = torch.randperm(trainData:size())
       trainData.shard[nid] = shffl[{ {1,shardSize[nid]} }]
    end
  end
  if opt.imbalanced==1 and opt.IID==1 then
    imbalanced_shardSizeFrac={}
    print('imbalanced shardSizeFrac for total '.. opt.netSize .. ' parties:')

    for i=1,opt.netSize do 
      imbalanced_shardSizeFrac[i]=torch.uniform(0.1, 0.9)
    end
    local frac_sum=0
    for i = 1, opt.netSize do
        frac_sum = frac_sum + imbalanced_shardSizeFrac[i]
    end
    for i=1,opt.netSize do 
      imbalanced_shardSizeFrac[i]=imbalanced_shardSizeFrac[i]/frac_sum
      print(imbalanced_shardSizeFrac[i]) 
    end
    -- balanced: 600 each, total 600*opt.netSize, imbalanced partition among opt.netSize
    total_records=math.ceil(opt.shardSizeFrac * trainSize * opt.netSize)
    for nid = 1, opt.netSize do
      shardSize[nid] = math.ceil(imbalanced_shardSizeFrac[nid] * total_records)
      print('imbalanced shardSize for party '.. nid .. ': ' .. shardSize[nid] ..'\n')
      local shffl = torch.randperm(trainData:size())
      trainData.shard[nid] = shffl[{ {1,shardSize[nid]} }]
    end
  end
  -- non IID data shard 
  if opt.IID==0 then
    train_class_order_file=paths.concat(opt.folder, 'train_class_order.' .. opt.shardID .. '.' .. opt.run) 
    train_class_num_file=paths.concat(opt.folder, 'train_class_num.' .. opt.shardID .. '.' .. opt.run) 
    print('non-IID data shard')
    local class_len=10
    class_indices={}
    for i=1,class_len do
      -- jth raw record corresponds to indices th item with label i
      class_indices[i]={}
      local indices=0
      for j=1,trainSize do
        if trainData.labels[j]==i then
          indices=indices+1
          class_indices[i][indices]=j
        end
      end
    end
    for nid = 1, opt.netSize do
      local l=0
      shardSize[nid] = math.ceil(opt.shardSizeFrac * trainSize)
      trainData.shard[nid]=torch.FloatTensor(shardSize[nid]):zero()
      biased_class = torch.random(class_len)
      print(biased_class)
      biased_class_len = math.floor(shardSize[nid]*0.5)
      left_classes_len = math.floor(shardSize[nid]*0.5)
      shuff = torch.randperm(#class_indices[biased_class])
      for s=1,biased_class_len do
        l=l+1
        trainData.shard[nid][l] = class_indices[biased_class][shuff[s]]
      end
      left_classes_Frac={}
      for i=1,class_len do 
        if i==biased_class then
          left_classes_Frac[i]=0
        else
          left_classes_Frac[i]=torch.uniform(0.1, 0.9)
        end
      end
      local frac_sum=0
      for i = 1, class_len do
          frac_sum = frac_sum + left_classes_Frac[i]
      end
      if biased_class==class_len then
        last_class=class_len-1
      else
        last_class=class_len
      end
      left_class_len={}
      per_class_num={}
      local classes_len=0
      for i=1,class_len do 
        if i==biased_class then
          left_class_len[i]=0
          per_class_num[i]=biased_class_len
        else
          -- ensure each party has shardSize[nid] examples, by allocating more examples to the last left class
          if i==last_class then
            left_class_len[i]=left_classes_len-classes_len
          else
            left_class_len[i]=math.floor(left_classes_Frac[i]/frac_sum*left_classes_len)
            classes_len=classes_len+left_class_len[i]
          end
          per_class_num[i]=left_class_len[i]
          shuff = torch.randperm(#class_indices[i])
          for s=1,left_class_len[i] do
            l=l+1
            trainData.shard[nid][l] = class_indices[i][shuff[s]]
          end
        end
      end
      -- per_class_num: sort as per ascending order, improve generalisation after collaboration: per-class test acc vs class distribution
      train_class_num, train_class_order = torch.FloatTensor(per_class_num):abs():sort(1)
    end
    print(trainData.shard[1])
    print(trainData.shard[1][1])
    torch.save(train_class_order_file, train_class_order)
    torch.save(train_class_num_file, train_class_num)
  end
  torch.save(shardfile, trainData.shard)
end

print(trainData)
print(testData)

if opt.dataset == 'mnist' then
   nfeats = 1
   width  = 32
   height = 32
   classes = {'1','2','3','4','5','6','7','8','9','0'}

   if opt.model == 'deep' then
    --    60000
    --     32
    --     32
    -- [torch.LongStorage of size 3]
      trainData.data = trainData.data:squeeze()
      testData.data = testData.data:squeeze()
   end
end

ninputs   = nfeats * width * height
nhiddens  = 128 -- ninputs / 6
nhiddens2 = 64 -- ninputs / 12
noutputs  = 10

nstates = {64,64,128}
filtsize = 5
poolsize = 2
normkernel = image.gaussian1D(7)

-- constructing the model
model = nn.Sequential()

if opt.model == 'linear' then
  -- Simple linear model
  model:add(nn.Reshape(ninputs))
  model:add(nn.Linear(ninputs,noutputs))

elseif opt.model == 'mlp' then
  -- Simple 2-layer neural network, with tanh hidden units
  model:add(nn.Reshape(ninputs))
  model:add(nn.Linear(ninputs,nhiddens))
  model:add(nn.Tanh())
  model:add(nn.Linear(nhiddens,noutputs))

elseif opt.model == 'deep' then
  -- Deep neural network, with ReLU hidden units
  model:add(nn.Reshape(ninputs))
  model:add(nn.Linear(ninputs,nhiddens))
  model:add(nn.ReLU())
  model:add(nn.Linear(nhiddens,nhiddens2))
  model:add(nn.ReLU())
  model:add(nn.Linear(nhiddens2,noutputs))
elseif opt.model == 'cvn' then
   -- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
   model:add(nn.SpatialConvolutionMM(1, 32, 5, 5))
   model:add(nn.Tanh())
   model:add(nn.SpatialMaxPooling(3, 3, 3, 3))

   -- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
   model:add(nn.SpatialConvolutionMM(32, 64, 5, 5))
   model:add(nn.Tanh())
   model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

   -- stage 3 : standard 2-layer MLP:
   model:add(nn.Reshape(64*2*2))
   model:add(nn.Linear(64*2*2, 200))
   model:add(nn.Tanh())
   model:add(nn.Linear(200, noutputs))
end
-- define loss
model:add(nn.LogSoftMax())

-- printing the model
print(model)

criterion = nn.ClassNLLCriterion()

-- prepare for training
-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- parameters is different for each run
-- Retrieve parameters and gradients (1-dim vector)
parameters,gradParameters = model:getParameters()

print(parameters:nElement())

-- SGD optimizer
optimState = {
   momentum          = opt.momentum,
   weightDecay       = opt.weightDecay,
   learningRate      = opt.learningRate,
   learningRateDecay = opt.learningRateDecay
}
optimMethod = optim.sgd

epoch = 0
print('epoch: ' .. epoch)
 
function train(e,node)
   local time = sys.clock()
   -- set model to training mode
   model:training()
   -- shuffle at each epoch
   local shuffle = torch.randperm(shardSize[node])
   -- Final loss
   local final_loss
   -- do opt.epochFraction of one epoch
   for t = 1, math.ceil(opt.epochFraction * shardSize[node]), opt.batchSize do
      -- create mini batch
      local inputs = {}
      local targets = {}
      for i = t, math.min(t+opt.batchSize-1,shardSize[node]) do
         -- load new sample
         local inx = trainData.shard[node][shuffle[i]]
         local input = trainData.data[inx]
         local target = trainData.labels[inx]
         if opt.type == 'double' then input = input:double() end
         table.insert(inputs, input)
         table.insert(targets, target)
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end
                       -- reset gradients
                       gradParameters:zero()
                       -- f is the average of all criterions
                       local f = 0
                       -- evaluate function for complete mini batch
                       for i = 1,#inputs do
                          -- estimate f
                          -- inputs[i]:[torch.FloatTensor of size 32x32]
                          local output = model:forward(inputs[i])
                          local err = criterion:forward(output, targets[i])
                          f = f + err
                          -- estimate df/dW
                          local df_do = criterion:backward(output, targets[i])
                          model:backward(inputs[i], df_do)
                          -- update confusion
                          confusion:add(output, targets[i])
                       end
                       -- normalize gradients and f(X)
                       gradParameters:div(#inputs)
                       f = f/#inputs
                       final_loss = f
                       -- return f and df/dX
                       return f,gradParameters
                    end
      -- optimize on current mini-batch
      optimMethod(feval, parameters, optimState)
   end
   -- accuracy
   accuracy = confusion.mat:diag():sum() / confusion.mat:sum()
   -- time taken
   time = sys.clock() - time
   -- traintime = time * 1000 / trainData.shardsize()
   traintime = time * 1000 / shardSize[node]
   -- next epoch
   confusion:zero()
   return accuracy, final_loss
end

function test()
   -- local vars
   local time = sys.clock()
   -- set model to evaluate mode
   model:evaluate()   
   local f = 0
   -- test over test data
   for t = 1,testData:size() do
      -- get new sample
      local input = testData.data[t]
      if opt.type == 'double' then input = input:double() end
      local target = testData.labels[t]

      -- test sample
      local pred = model:forward(input)
      local err = criterion:forward(pred, target)
      f = f + err
      confusion:add(pred, target)
   end
   -- timing
   time = sys.clock() - time
   time = time / testData:size()
   accuracy = confusion.mat:diag():sum() / confusion.mat:sum()
   testtime = time * 1000
   -- next iteration:
   confusion:zero()
   return accuracy, f/testData:size()
end

function test_perclass()
   -- local vars
   local time = sys.clock()
   -- set model to evaluate mode
   model:evaluate()
   local f = 0
   accuracy={}
   test_class_indices = torch.load(test_classfile, 'binary')
   for i=1,class_len do
      test_inx=test_class_indices[i]
     -- test over per-class test data
     for t = 1,#test_inx do
        -- get new sample
        local input = testData.data[test_inx[t]]
        if opt.type == 'double' then input = input:double() end
        local target = testData.labels[test_inx[t]]
        -- test sample
        local pred = model:forward(input)
        local err = criterion:forward(pred, target)
        f = f + err
        confusion:add(pred, target)
     end
     accuracy[i] = confusion.mat:diag():sum() / confusion.mat:sum()
     confusion:zero()
   end
    -- timing
   time = sys.clock() - time
   time = time / testData:size()
   testtime = time * 1000
   return accuracy, f/testData:size()
end

function train_80(node,train_len,train_idx)
   -- local vars
   local time = sys.clock()
   -- set model to training mode
   model:training()
   for local_epoch = 1, opt.local_nepochs do
    print('local_epoch: '..local_epoch)
     -- shuffle at each epoch
     local shuffle = torch.randperm(train_len)
     -- Final loss
     local final_loss
     -- do opt.epochFraction of one epoch
     for t = 1, math.ceil(opt.epochFraction * train_len), opt.batchSize do
        -- create mini batch
        local inputs = {}
        local targets = {}
        for i = t, math.min(t+opt.batchSize-1,train_len) do
           local inx = train_idx[shuffle[i]]
           local input = trainData.data[inx]
           local target = trainData.labels[inx]
           if opt.type == 'double' then input = input:double() end
           table.insert(inputs, input)
           table.insert(targets, target)
        end
        -- create closure to evaluate f(X) and df/dX
        local feval = function(x)
                         -- get new parameters
                         if x ~= parameters then
                            parameters:copy(x)
                         end
                         -- reset gradients
                         gradParameters:zero()
                         -- f is the average of all criterions
                         local f = 0
                         -- evaluate function for complete mini batch
                         for i = 1,#inputs do
                            -- estimate f
                            local output = model:forward(inputs[i])
                            local err = criterion:forward(output, targets[i])
                            f = f + err
                            -- estimate df/dW
                            local df_do = criterion:backward(output, targets[i])
                            model:backward(inputs[i], df_do)
                            -- update confusion
                            confusion:add(output, targets[i])
                         end
                         -- normalize gradients and f(X)
                         gradParameters:div(#inputs)
                         f = f/#inputs
                         final_loss = f
                         -- return f and df/dX
                         return f,gradParameters
                      end
        -- optimize on current mini-batch
        optimMethod(feval, parameters, optimState)
     end
   end
   -- accuracy
   accuracy = confusion.mat:diag():sum() / confusion.mat:sum()
   -- time taken
   time = sys.clock() - time
   traintime = time * 1000 / train_len
   -- next epoch
   confusion:zero()
   -- epoch = epoch + 1
   return accuracy, final_loss
end

-- return pred for Gan samples
function test_gan( len, samples )
   local time = sys.clock()
   -- set model to evaluate mode
   model:evaluate()
   local f = 0
   -- test over j's sample data
   local test_preds = {}
   for t = 1, len do
      -- get new sample
      local input = samples[t]
      if opt.type == 'double' then input = input:double() end
      -- test sample
      local pred = model:forward(input)
      if pred:nDimension()==1 then
        _, predicted = torch.max(pred, 1)
      else
        row=pred:size()[1]
        column=pred:size()[2]
        -- [torch.FloatTensor of size nx1]
        if row>column then
          _, predicted = torch.max(pred, 1)
        else
        -- [torch.FloatTensor of size 1xn]
          _, predicted = torch.max(pred, 2)
        end
      end
      test_preds[t]= predicted
   end

   -- timing
   time = sys.clock() - time
   time = time / len
   testtime = time * 1000
   return test_preds
end