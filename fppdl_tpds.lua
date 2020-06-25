dofile "local_sgd.lua"
require 'csvigo'
require 'image'
require 'paths'

if opt.imbalanced==1 then
  if epoch==0 then
    file = io.open("logs/fppdl_"..opt.dataset.."_"..opt.model.."_p"..opt.netSize.."e"..opt.nepochs.."_slevel0"..opt.slevel.."_imbalanced_".. "IID".. opt.IID .. "_pretrain".. opt.pretrain .. "_localepoch".. opt.local_nepochs .. "_localbatch".. opt.batchSize .. "_lr".. opt. learningRate .. "_" .. opt.run .."_tpds.log", "w")
  else
    file = io.open("logs/fppdl_"..opt.dataset.."_"..opt.model.."_p"..opt.netSize.."e"..opt.nepochs.."_slevel0"..opt.slevel.."_imbalanced_".. "IID".. opt.IID .. "_pretrain".. opt.pretrain .. "_localepoch".. opt.local_nepochs .. "_localbatch".. opt.batchSize .. "_lr".. opt. learningRate .. "_" .. opt.run .."_tpds.log", "a")
  end
else
  if epoch==0 then
    file = io.open("logs/fppdl_"..opt.dataset.."_"..opt.model.."_p"..opt.netSize.."e"..opt.nepochs.."_slevel0"..opt.slevel.."_IID".. opt.IID .. "_pretrain".. opt.pretrain.. "_localepoch".. opt.local_nepochs .. "_localbatch".. opt.batchSize .. "_lr".. opt. learningRate .. "_" .. opt.run .."_tpds.log", "w")
  else
    file = io.open("logs/fppdl_"..opt.dataset.."_"..opt.model.."_p"..opt.netSize.."e"..opt.nepochs.."_slevel0"..opt.slevel.."_IID".. opt.IID .. "_pretrain".. opt.pretrain.. "_localepoch".. opt.local_nepochs .. "_localbatch".. opt.batchSize .. "_lr".. opt. learningRate .. "_" .. opt.run .."_tpds.log", "a")
  end
end
io.output(file)
----------------------------------------------------------------
function normalize(t_i, i)
   local z = 0.0
   t_i[i] = 0
   for j = 1, opt.netSize do
      z = z + t_i[j]
   end
   for j = 1, opt.netSize do
      t_i[j] = t_i[j] / z
   end
   return t_i
end

----------------------------------------------------------------
function fileExists(path)
    local file = io.open(path, "r")
    if file then
        io.close(file)
        return true
    end
    return false
end
----------------------------------------------------------------
local nid
-- initially sample data points : should be done by DPGANs. assuming 1 bugdet point for 1 parameter
print("phase 1")
local train_acc = {}
local test_acc = {}
local share_level = {}
local point = {}
local credit = {}
local iacc = {}
local sacc = {}
local train_samples={}
local train_samples_random={}
local ps = {} -- parameters of all nodes for standalone
local ps_pretrain = {} -- each node train 5/10 epochs before collaboration
if paths.filep(psfile) and paths.filep(pspretrain_file) then
  ps = torch.load(psfile, 'binary')
  ps_pretrain = torch.load(pspretrain_file, 'binary')
else
  -- same initialization for all parties to begin collaboration, or for standalone training
   for nid = 1, opt.netSize do
      ps[nid] = parameters * 0.0
      ps[nid]:copy(parameters)
      ps_pretrain[nid] = parameters * 0.0
      ps_pretrain[nid]:copy(parameters)
   end
end
for nid = 1, opt.netSize do
	train_samples[nid] = {}
	train_samples_random[nid] = {}
end

if opt.IID==1 then
  sample_path=opt.dataset..'_p'..opt.netSize..'e100_slevel01'..'_imbalanced'..opt.imbalanced..'_dpgansamples_'..opt.run
else
	sample_path=opt.dataset..'_IID0_p'..opt.netSize..'e100_slevel0'..opt.slevel..'_imbalanced'..opt.imbalanced..'_dpgansamples_'..opt.run
end

party_path = io.open ('./'..opt.dataset..'_party'..opt.netSize..'_imbalanced'..opt.imbalanced, "w")
io.output(party_path)
io.write(sample_path)
io.close(party_path)
io.output(file)
if not paths.dirp(sample_path) then
	os.execute('mkdir ' .. sample_path)
end
-- sample_len: how many figs are generated
local sample_len = {}
for nid = 1, opt.netSize do
	if opt.slevel==1 then
    	share_level[nid] = 0.1
	else 
	  if paths.filep(slevelfile) then
	    print('load slevel')
	    share_level = torch.load(slevelfile, 'binary')
	  else
	    print('allocate slevel')
	    if opt.slevel==5 then
	     -- share_level[nid] = torch.uniform(0.1, 0.5) --magic no. should be controled by hyper-p
       share_level[nid] = torch.uniform(0.1, 1)
	    end
	  end
	end 
	point[nid] = math.floor( parameters:nElement() * share_level[nid] * (opt.netSize-1) )
	sample_len[nid] = math.floor( shardSize[nid] * share_level[nid] )     
  if sample_len[nid] < 1 then -- assuming each party at least release one sample
      sample_len[nid] = 1
  end
  io.write("party " .. nid .. " share level: " .. share_level[nid],'\n')
  io.write("party " .. nid .. " sample_len: " .. sample_len[nid] .. " in each epoch",'\n')
  io.write("party " .. nid .. " gets " .. point[nid] .. " points from sampling.",'\n')
  -- call tf gan to generate samples
  fig_path = sample_path..'/party'..nid..'_samples/'..'samples.png'
  print(fig_path)
  if not fileExists(fig_path) then
    print('generate samples from the beginning')
    party_indices={}
      for t = 1, shardSize[nid] do
        local inx = trainData.shard[nid][t]
          -- py starts at 0, rather than 1 in torch
         table.insert(party_indices,inx-1)
      end
    indices_path = paths.concat(sample_path, 'party_indices')
    csvigo.save{path=indices_path,data={party_indices}}
    num_samples_path = paths.concat(sample_path, 'num_samples')
    csvigo.save{path=num_samples_path,data={{sample_len[nid]*opt.nepochs}}}
    nid_path = paths.concat(sample_path, 'nid')
    csvigo.save{path=nid_path,data={{nid}}}
    ltn12 = require('ltn12')
    if opt.dataset == 'mnist' then
       lfs=require 'lfs'
       rootPath = lfs.currentdir()
       lfs.chdir('dpgan')
       os.execute('python dp_mnist.py -a --epsilon 4.0 --target-epsilon 4.0 --data-dir ./data/mnist_data --sample-ratio 0.02 -g 4 --clipper mnist_est --batch-size 64 --save-every 500 --image-every 50 --adaptive-rate --num-critic 4 --num-epoch 100 --terminate --has_gpu 1 '..'--party '..opt.netSize..' --imbalanced '..opt.imbalanced)
       lfs.chdir(rootPath)
    end
  else
    print('load samples, total sample no:')
    print(sample_len[nid])
  end
  -- fig range:[0-1]
  if opt.dataset == 'mnist' then
    fig_path = sample_path..'/party'..nid..'_samples/'..'samples.png'
    fig=image.load(fig_path)
    fig1=fig:reshape(fig:size(1)*fig:size(2)*fig:size(3)/28/28,1,28,28)
    for s=1,fig1:size(1) do
      -- reshape tf fig:28*28 tensor to torch 32*32 tensor
      a=torch.Tensor(1,28,2):fill(0)
      sample_tensor=torch.cat(a,fig1[s],3)
      sample_tensor=torch.cat(sample_tensor,a,3)
      b=torch.Tensor(1,2,32):fill(0)
      sample_tensor=torch.cat(b,sample_tensor,2)
      sample_tensor=torch.cat(sample_tensor,b,2)
      train_samples[nid][s]=sample_tensor
    end
  end
end
if opt.slevel==5 then
  torch.save(slevelfile,share_level, 'binary')
end
----------------------------------------------------------------
-- pre-train aprior model
for nid = 1, opt.netSize do
  iacc[nid] = 0
  sacc[nid] = 0
  train_acc[nid] = 0
  test_acc[nid] = 0
end
-- each party evaluate other parties using its pre-trained aprior model
if opt.IID==0 then
  sorted_sacc={}
  sorted_max_sacc={}
  for c=1,class_len do
    sorted_max_sacc[c]=0
  end
end
for nid = 1, opt.netSize do
  if paths.filep(psfile) then
    print("\nload pre-trained aprior models")
    parameters:copy(ps[nid])
  else
    print("\nPre-train aprior models")
    parameters:copy(ps[nid])
    for e = epoch+1,opt.nepochs do
      accuracy, loss = train(e,nid)
      if e==opt.pretrain_epochs then
        ps_pretrain[nid]:copy(parameters)
      end
    end
    -- standalone model
    ps[nid]:copy(parameters) 
  end
  if opt.IID==1 then
    accuracy, loss = test()
  else
    accuracy, loss = test_perclass()
  end
  sacc[nid] = accuracy
  if opt.IID==1 then
    io.write('standalone party ' .. nid .. ' test acc ' .. sacc[nid],'\n')
  else
    print('standalone party ' .. nid .. ' test acc ','\n')
    print(torch.Tensor(sacc[nid]))
    sorted_sacc[nid]=torch.sort(torch.Tensor(sacc[nid]))
  end
end
-- ps at opt.nepochs
if not paths.filep(psfile) and not paths.filep(pspretrain_file) then
  torch.save(psfile, ps, 'binary')
  torch.save(pspretrain_file, ps_pretrain, 'binary')
end
if opt.IID==0 then
  for c=1,class_len do
    for nid = 1, opt.netSize do
      sorted_max_sacc[c]=math.max(sorted_max_sacc[c],sorted_sacc[nid][c])
    end
  end
  print('standalone party max test acc as per ascending order of class distribution:')
  print(torch.Tensor(sorted_max_sacc))
end
-----------------------------------------------------------------
-- a simple way for majority voting
local class_len=10
for i = 1, opt.netSize do 
  local sample_shuffle = torch.randperm(#train_samples[i])
	for s=1,sample_len[i] do
    train_samples_random[i][s] = train_samples[i][sample_shuffle[s]]
  end
  credit[i] = {}
  local preds_tensor = torch.FloatTensor()
  local pred = {}
  local majority_class_loc = {}
  local majority_class_combined = {}
  local combined_labels={}
  for j = 1, opt.netSize do 
    -- each party j test samples of party i
    parameters:copy(ps[j])
    pred[j]={}
    io.write(i,',',j,'\n')
    pred[j] = test_gan(sample_len[i], train_samples_random[i])
    local preds_j = torch.FloatTensor()
    for k = 1, sample_len[i] do
      preds_j=torch.cat(preds_j,pred[j][k]:type('torch.FloatTensor'),1)
    end
    preds_tensor=torch.cat(preds_tensor,preds_j,2)
  end
  -- count label frequency, and record the label with quorum
  frequency={}
  for k=1,sample_len[i] do
    frequency[k]={}
    for c=1,class_len do
      frequency[k][c] = preds_tensor[k]:eq(c):sum()
    end
    for c=1,class_len do
      if frequency[k][c]==math.max(table.unpack(frequency[k])) then
        majority_class_combined[k]=c
      end  
    end
  end

  for j = 1, opt.netSize do 
    if j~=i then
      correct_labels=0
      for k=1,sample_len[i] do
        if torch.FloatTensor({majority_class_combined[k]})[1]==preds_tensor:index(2,torch.LongTensor({j}))[k][1] then
          correct_labels=correct_labels+1
        end
      end
      credit[i][j] = correct_labels/sample_len[i]
    else
      credit[i][j] = 0
    end
  end
  credit[i] = normalize( credit[i], i )
end

for i = 1, opt.netSize do
  for j = 1, opt.netSize do
    io.write('\nparty ' .. j .. ' initial credit given by party ' .. i .. ': ' .. credit[i][j],'\n')
  end
end

credible_set = {}
for j = 1, opt.netSize do
  credible_set[j]=j
end
if opt.credit_thres==1 then 
  credit_threshold=1/3*1/(#credible_set-1)
end
-- if most parties vote party j as low-contribution, party j will be removed from system
for i = 1, opt.netSize do
  -- count how many party vote i as low-contribution
  credible_vote=0
  for j = 1, opt.netSize do
    if j~=i then
      if credit[j][i] < credit_threshold then
        credible_vote=credible_vote+1
      end
    end
  end
  if credible_vote>=math.ceil(2/3*#credible_set) then
    credible_set[i]=nil
  end
end
if #credible_set<=2 then
  io.write('less than 2 parties left, break! 2nd will not start!')
else
  -----------------------------------------------------------------
  -- track the total number of downloaded gradients for each party during training, to quantify fairness
  local uploads_n = {} -- uploads of all nodes
  local downloads_n = {}
  for i = 1, opt.netSize do
    downloads_n[i] = 0
    uploads_n[i] = 0
  end
  -----------------------------------------------------------------
  -- second phase:
  -- each party trains its local model, and spend points to download others' gradients.
  print("\nphase 2")
  local max_acc=0
  train_idx={}
  train_len={}
  for nid = 1, opt.netSize do
    shuff = torch.randperm(shardSize[nid])
    train_len[nid]= math.ceil(shardSize[nid])
    io.write('train_len: ' .. train_len[nid],'\n')
    train_idx[nid] =trainData.shard[nid][{ shuff[{ {1,train_len[nid]} }] }]
  end
  local g = {} -- gradeints of all nodes
  local g_o = {} -- gradeints of all nodes
  for nid = 1, opt.netSize do
      g[nid] = parameters * 0.0
      -- at epoch t+1, each party should combine gradiets of epoch t from other parties 
      g_o[nid] = parameters * 0.0
  end

  local p = {} -- parameters of all nodes for collaboration
  if opt.pretrain==1 then
   for nid = 1, opt.netSize do
      p[nid] = parameters * 0.0 
      p[nid]:copy(ps_pretrain[nid])
      io.write('load pre-trained paras of 10 epochs','\n')
   end
  else
    init_parameters= parameters * 0.0
    init_parameters:copy(parameters)
    for nid = 1, opt.netSize do
      p[nid] = parameters * 0.0
      p[nid]:copy(init_parameters)
      io.write('load same init parameters','\n')
    end
  end

  for e = epoch+1,opt.nepochs do
    io.write('credit threshold in epoch ' .. e .. ' ' .. credit_threshold,'\n')
    local max_acc_fppdl=0
     -- in each epoch: train local models based on its own data, integrate all its paras, then download 10% from other parties
    for nid = 1, opt.netSize do
      parameters:copy(p[nid])
      accuracy, loss = train_80(nid,train_len[nid],train_idx[nid])
      train_acc[nid]=accuracy
      io.write('train\tepoch ' .. e .. '\tparty ' .. nid .. '\t acc ' .. train_acc[nid],'\n')
      accuracy, loss = test()
      test_acc[nid] = accuracy
      io.write('test\tepoch ' .. e .. '\tparty ' .. nid .. '\t acc ' .. test_acc[nid],'\n')
      if e==1 then
        iacc[nid] = test_acc[nid]
      end
      -- bound gradient to remove noise
      g[nid]:copy(parameters - p[nid])
      if opt.range > 0 then
        g[nid]:apply(function(x) return cut(x, opt.range) end)
      end
      -- integrating all its own gradients with rangng cut
      for elmnt = 1, parameters:nElement() do
        p[nid][elmnt] = p[nid][elmnt] + g[nid][elmnt] -- update the parameters of i based on j's parameters, given the threshold of top-k parameters
      end
      -- gradients of current epoch:synchronous protocol
      g_o[nid]:copy(g[nid])
    end

     -- use points of last epoch for transaction in current epoch  
    local point_n = {}
    for i = 1, opt.netSize do
      point_n[i] = point[i]
    end
     
     -- download other 10%
    for i = 1, opt.netSize do
      if point[i] <= 0 then
        print("party " .. i .. " died due to no point")
      else
         -- download gradients from party j
        for j = 1, opt.netSize do
          if j~=i then
            -- keep record of gradients downloaded from party j
            if tonumber(credit[i][j]) >= credit_threshold then
             local download_num = math.floor(credit[i][j] * point[i])
             if download_num > math.floor(parameters:nElement() * share_level[j]) then -- consider j's share_level
               download_num = math.floor(parameters:nElement() * share_level[j])
             end
             
             if download_num > 0 then
                downloads_n[i] = downloads_n[i]+download_num
                uploads_n[j] = uploads_n[j]+download_num
                -- update point
                point_n[i] = point_n[i] - download_num
                point_n[j] = point_n[j] + download_num           
                local delta = parameters * 0.0
                local temp_delta = parameters * 0.0
                delta:copy(g_o[j])
                if opt.update_criteria=='large' then
                  temp_delta:copy(delta)
                  -- largest values gradients, from smallest->largest
                  _, inx = temp_delta:abs():sort(1)
                  threshold = temp_delta[ inx[-download_num] ]
                else
                  -- random threshold
                  threshold=0.0001
                end
                 io.write('downloaded gradient threshold ' .. threshold,'\n')

                local perm = torch.randperm(parameters:nElement())
                local count = 0
                upval = 0
                for elmnt = 1, parameters:nElement() do
                  local eid = perm[elmnt] 
                  if math.abs(delta[eid]) >= threshold then
                    upval = delta[eid]
                    p[i][eid] = p[i][eid] + (upval / 1.0) -- update the parameters of i based on j's parameters, given the threshold of top-k parameters
                    p[i][eid] = p[i][eid] + (upval*credit[i][j] / math.max(table.unpack(credit[i])))
                    count = count + 1
                    if count >= download_num then
                      break
                    end
                  end
                end
              end
            end
          end
        end
       -- print budget of i
       io.write("party " .. i .. " now has " .. point_n[i] .. " points.",'\n')
      end
    end
     -- update points after each epoch: each party uses previous epoch's points for transaction
    for i = 1, opt.netSize do
      point[i] = point_n[i]
      max_acc_fppdl=math.max(max_acc_fppdl,test_acc[i])
    end
    io.write('in epoch '.. e .. ', max fppdl test acc '.. max_acc_fppdl,'\n')
    max_acc=math.max(max_acc,max_acc_fppdl)
    -----------------------------------------------------------------
    -- update credit: majority voting
    for i = 1, opt.netSize do 
      local sample_shuffle = torch.randperm(#train_samples[i])
    	for s=1,sample_len[i] do
        train_samples_random[i][s] = train_samples[i][sample_shuffle[s]]
  	  end
      local preds_tensor = torch.FloatTensor()
      local pred = {}
      local majority_class_loc = {}
      local majority_class_combined = {}
      local combined_labels={}
      for j = 1, opt.netSize do 
        -- each party j test samples of party i using current model
        parameters:copy(p[j])
        pred[j]={}
        pred[j] = test_gan(sample_len[i], train_samples_random[i])
        local preds_j = torch.FloatTensor()
        for k = 1, sample_len[i] do
          preds_j=torch.cat(preds_j,pred[j][k]:type('torch.FloatTensor'),1)
        end
        preds_tensor=torch.cat(preds_tensor,preds_j,2)
      end
      -- count label frequency, and record the label with quorum
      frequency={}
      for k=1,sample_len[i] do
        frequency[k]={}
        for c=1,class_len do
          frequency[k][c] = preds_tensor[k]:eq(c):sum()
        end
        for c=1,class_len do
          if frequency[k][c]==math.max(table.unpack(frequency[k])) then
            majority_class_combined[k]=c
          end  
        end
      end

      for j = 1, opt.netSize do 
        if j~=i then
          correct_labels=0
          for k=1,sample_len[i] do
            if torch.FloatTensor({majority_class_combined[k]})[1]==preds_tensor:index(2,torch.LongTensor({j}))[k][1] then
              correct_labels=correct_labels+1
            end
          end
          if opt.credit_fade==1 then
            current_credit = correct_labels/sample_len[i]
            credit[i][j] = credit[i][j]*0.2+current_credit*0.8
          else
            credit[i][j] = (credit[i][j]+current_credit)/2
          end
        else
        	credit[i][j] = 0
        end
      end
      credit[i] = normalize( credit[i], i )
    end
    for i = 1, opt.netSize do
      for j = 1, opt.netSize do
        io.write('\nparty ' .. j .. ' updated credit given by party ' .. i .. ': ' .. credit[i][j],'\n')
      end
    end
    -- if most parties vote party i as low-contribution, party i will be removed from system
    for i = 1, opt.netSize do
      -- count how many party vote i as low-contribution
      credible_vote=0
      for j = 1, opt.netSize do
        if j~=i then
          if credit[j][i] < credit_threshold then
            credible_vote=credible_vote+1
          end
        end
      end
      if credible_vote>=math.ceil(2/3*opt.netSize) then
        credible_set[i]=nil
      end
    end
    credit_threshold=1/3*1/(#credible_set-1)
    print('credible_set in epoch ' .. e .. ' ')
    print(credible_set)
    if #credible_set<=2 then
      io.write('less than 2 parties left, break collaboration!','\n')
      break
    end
  end 
  ----------------------------------------------------------------
  -- fairness computation
  for i = 1, opt.netSize do
    downloads_n[i]=downloads_n[i]/opt.nepochs
    uploads_n[i]=uploads_n[i]/opt.nepochs
    io.write('party ' .. i .. ' final test acc ' .. test_acc[i],'\n')
    io.write('party ' .. i .. ' downloaded ' .. downloads_n[i] .. ' gradients in average','\n')
    io.write('party ' .. i .. ' uploaded ' .. uploads_n[i] .. ' gradients in average','\n')
    -- final points
    io.write('party ' .. i .. ' cumulative points ' .. point[i],'\n')
  end
  io.write('in all epochs ' .. ', max fppdl test acc '.. max_acc)
  -- closes the open file
  io.close(file)
end   