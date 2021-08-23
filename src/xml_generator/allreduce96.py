import math
instances = 4
ngpus = 4
nchunksperloop = instances*ngpus
print(f'<algo name="allreduce_small" nchunksperloop="{nchunksperloop}" nchannels="{instances}" proto="Simple" ngpus="{ngpus}">')

for i in range(ngpus):
    tbindex = 0
    print(f'  <gpu id="{i}" i_chunks="{nchunksperloop}" o_chunks="{nchunksperloop}" s_chunks="{nchunksperloop}">')
    for ch in range(instances//2):
      print(f'    <tb id="{tbindex}" send="{(i+1)%ngpus}" recv="{(i-1)%ngpus}" chan="{ch}">')
      step = 0
      oldnghr = i
      for j in range(ngpus):
        nghr = (i-j+ngpus)%ngpus
        oldnghr = nghr
        if j > 0:
          print(f'      <step s="{step}" type="rrc" srcbuf="i" srcoff="{nghr*instances+ch}" dstbuf="i" dstoff="{nghr*instances+ch}" cnt="1" depid="-1" deps="-1" hasdep="0"/>')
          step += 1
        if j < ngpus-1:
          print(f'      <step s="{step}" type="s" srcbuf="i" srcoff="{oldnghr*instances+ch}" dstbuf="i" dstoff="{oldnghr*instances+ch}" cnt="1" depid="-1" deps="-1" hasdep="0"/>')
          step += 1
      for j in range(ngpus):
        nghr = (i-j+1+ngpus)%ngpus
        oldnghr = nghr
        if j > 0:
          print(f'      <step s="{step}" type="r" srcbuf="i" srcoff="{oldnghr*instances+ch}" dstbuf="i" dstoff="{oldnghr*instances+ch}" cnt="1" depid="-1" deps="-1" hasdep="0"/>')
          step += 1
        if j < ngpus-1:
          print(f'      <step s="{step}" type="s" srcbuf="i" srcoff="{nghr*instances+ch}" dstbuf="i" dstoff="{nghr*instances+ch}" cnt="1" depid="-1" deps="-1" hasdep="0"/>')
          step += 1

      print('    </tb>')
      tbindex+=1
    for ch in range(instances//2,instances):
      print(f'    <tb id="{tbindex}" send="{(i-1)%ngpus}" recv="{(i+1)%ngpus}" chan="{ch}">')
      step = 0
      oldnghr = i
      for j in range(ngpus):
        nghr = (i+j+ngpus)%ngpus
        oldnghr = nghr
        if j > 0:
          print(f'      <step s="{step}" type="rrc" srcbuf="i" srcoff="{nghr*instances+ch}" dstbuf="i" dstoff="{nghr*instances+ch}" cnt="1" depid="-1" deps="-1" hasdep="0"/>')
          step += 1
        if j < ngpus-1:
          print(f'      <step s="{step}" type="s" srcbuf="i" srcoff="{oldnghr*instances+ch}" dstbuf="i" dstoff="{oldnghr*instances+ch}" cnt="1" depid="-1" deps="-1" hasdep="0"/>')
          step += 1
      for j in range(ngpus):
        nghr = (i+j+1+ngpus)%ngpus
        oldnghr = nghr
        if j > 0:
          print(f'      <step s="{step}" type="r" srcbuf="i" srcoff="{oldnghr*instances+ch}" dstbuf="i" dstoff="{oldnghr*instances+ch}" cnt="1" depid="-1" deps="-1" hasdep="0"/>')
          step += 1
        if j < ngpus-1:
          print(f'      <step s="{step}" type="s" srcbuf="i" srcoff="{nghr*instances+ch}" dstbuf="i" dstoff="{nghr*instances+ch}" cnt="1" depid="-1" deps="-1" hasdep="0"/>')
          step += 1

      print('    </tb>')
      tbindex+=1
    print('  </gpu>')
print('</algo>')
