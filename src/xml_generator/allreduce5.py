import math
instances = 8
ngpus = 8
nnodes = 2
nchunksperloop = instances
print(f'<algo name="allreduce_small" nchunksperloop="{nchunksperloop}" nchannels="{instances*2}" proto="Simple" ngpus="{ngpus*nnodes}">')

ring = [1,3,2,6,7,5,4,0]
# ring = [0,1,2]

for n in range(nnodes):
  for i in range(ngpus):
    tbindex = 0
    print(f'  <gpu id="{n*ngpus+i}" i_chunks="{nchunksperloop}" o_chunks="{nchunksperloop}" s_chunks="{0}">')
    for ch in range(instances):
      sendpeer = -1
      recvpeer = -1
      for rr in range(ngpus):
        if ring[rr] == i:
          sendpeer = (n*ngpus+ring[rr+1]) if rr < ngpus-1 else ((1-n)*ngpus+ring[0])
          recvpeer = (n*ngpus+ring[rr-1]) if rr > 0 else ((1-n)*ngpus+ring[ngpus-1])
      print(f'    <tb id="{tbindex}" send="{sendpeer}" recv="{-1 if i == ring[0] else recvpeer}" chan="{ch}">')
      step = 0
      if i == ring[0]:
        t = "s"
      elif i == ring[ngpus-1]:
        t = "rrcs"
      else:
        t = "rrs"
      print(f'      <step s="{step}" type="{t}" srcbuf="i" srcoff="{ch}" dstbuf="i" dstoff="{ch}" cnt="{1}" depid="-1" deps="-1" hasdep="1"/>')
      step += 1
      print('    </tb>')
      step = 0
      tbindex+=1
      if ring[0] == i:
        print(f'    <tb id="{tbindex}" send="{-1}" recv="{recvpeer}" chan="{ch}">')
        print(f'      <step s="{step}" type="r" srcbuf="i" srcoff="{ch}" dstbuf="i" dstoff="{ch}" cnt="{1}" depid="{tbindex-1}" deps="0" hasdep="1"/>')

        print('    </tb>')
        tbindex+=1


      for rr in range(ngpus):
        if ring[rr] == i:
          sendpeer = (n*ngpus+ring[(rr+1)%ngpus])
          recvpeer = (n*ngpus+ring[(rr-1)%ngpus])
      print(f'    <tb id="{tbindex}" send="{recvpeer}" recv="{sendpeer}" chan="{ch+instances}">')
      step = 0
      if i == ring[0]:
          print(f'      <step s="{step}" type="s" srcbuf="i" srcoff="{ch}" dstbuf="i" dstoff="{ch}" cnt="{1}" depid="{tbindex-1}" deps="0" hasdep="0"/>')
          step += 1
          print(f'      <step s="{step}" type="r" srcbuf="i" srcoff="{ch}" dstbuf="i" dstoff="{ch}" cnt="{1}" depid="-1" deps="-1" hasdep="0"/>')
      else:
          t = "rrcs" if i == ring[ngpus-1] else "rcs"
          print(f'      <step s="{step}" type="{t}" srcbuf="i" srcoff="{ch}" dstbuf="i" dstoff="{ch}" cnt="{1}" depid="{tbindex-1}" deps="0" hasdep="0"/>')
      step += 1
      print('    </tb>')
      tbindex+=1

    print('  </gpu>')
print('</algo>')
