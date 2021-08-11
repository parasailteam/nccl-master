import math
instances = 1
ngpus = 8

nchunksperloop = instances*ngpus**2
print(f'<algo name="allreduce_small" nchunksperloop="{nchunksperloop}" nchannels="{instances}" proto="LL" ngpus="{ngpus}">')

for i in range(ngpus):
    tbindex = 0
    print(f'  <gpu id="{i}" i_chunks="{nchunksperloop}" o_chunks="{nchunksperloop}" s_chunks="{nchunksperloop}">')
    for j in range(ngpus):
            for ch in range(instances):
                print(f'    <tb id="{tbindex}" send="{j if i != j else -1}" recv="{j if i != j else -1}" chan="{ch}">')
                step = 0
                if i != j:
                    print(f'      <step s="{step}" type="s" srcbuf="i" srcoff="{j*instances*ngpus+ch}" dstbuf="s" dstoff="{i*instances*ngpus+ch}" cnt="{ngpus}" depid="-1" deps="-1" hasdep="0"/>')
                    step += 1
                    print(f'      <step s="{step}" type="r" srcbuf="i" srcoff="{i*instances*ngpus+ch}" dstbuf="s" dstoff="{j*instances*ngpus+ch}" cnt="{ngpus}" depid="-1" deps="-1" hasdep="1"/>')
                    step += 1
                for k in range(ngpus):
                    if k != i:
                        print(f'      <step s="{step}" type="re" srcbuf="s" srcoff="{k*instances*ngpus+ch+j}" dstbuf="i" dstoff="{i*instances*ngpus+ch+j}" cnt="{1}" depid="{k}" deps="1" hasdep="{1 if step == ngpus or (step == ngpus-2 and i == j) else 0}"/>')
                        step += 1
                if i != j:
                    for k in range(ngpus):
                        print(f'      <step s="{step}" type="nop" srcbuf="i" srcoff="0" dstbuf="i" dstoff="0" cnt="0" depid="{k}" deps="{ngpus if k != i else ngpus-2}" hasdep="{1 if step == ngpus else 0}"/>')
                        step += 1
                    print(f'      <step s="{step}" type="s" srcbuf="i" srcoff="{i*instances*ngpus+ch}" dstbuf="i" dstoff="{i*instances*ngpus+ch}" cnt="{ngpus}" depid="{-1}" deps="{-1}" hasdep="0"/>')
                    step += 1
                    print(f'      <step s="{step}" type="r" srcbuf="i" srcoff="{j*instances*ngpus+ch}" dstbuf="i" dstoff="{j*instances*ngpus+ch}" cnt="{ngpus}" depid="-1" deps="-1" hasdep="0"/>')
                    step += 1
                print('    </tb>')
                tbindex+=1
    print('  </gpu>')
print('</algo>')
