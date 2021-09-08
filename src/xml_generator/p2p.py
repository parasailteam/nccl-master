instances = 2
ngpus = 16
ngpuspernode = ngpus//2
sendFrom0 = 1*instances
nchunksperloop = (ngpuspernode-1)*instances+sendFrom0
print(f'<algo name="test" nchunksperloop="{nchunksperloop}" nchannels="{instances}" proto="Simple" ngpus="{ngpus}" redop="nop">')
for i in range(ngpus):
    tbindex = 0
    offset = sendFrom0-instances
    print('  <gpu id="{}" i_chunks="{}" o_chunks="{}" s_chunks="{}">'.format(i, nchunksperloop, nchunksperloop, instances))
    for ch in range(instances):
        if i == 0:
            print(f'    <tb id="{tbindex}" send="{ngpuspernode}" recv="-1" chan="{ch}">')
            print(f'      <step s="0" type="s" srcbuf="i" srcoff="{(sendFrom0//instances)*ch}" dstbuf="i" dstoff="{(sendFrom0//instances)*ch}" cnt="{sendFrom0//instances}" depid="-1" deps="-1" hasdep="0"/>')
            print(f'    </tb>')
            tbindex+=1
            for j in range(1,ngpuspernode):
                print(f'    <tb id="{tbindex}" send="{j}" recv="-1" chan="{ch}">')
                print(f'      <step s="0" type="s" srcbuf="i" srcoff="{offset+j*instances+ch}" dstbuf="s" dstoff="{ch}" cnt="1" depid="-1" deps="-1" hasdep="0"/>')
                print(f'    </tb>')
                tbindex+=1
        if i == ngpuspernode:
            print(f'    <tb id="{tbindex}" send="-1" recv="{0}" chan="{ch}">')
            print(f'      <step s="0" type="r" srcbuf="i" srcoff="{(sendFrom0//instances)*ch}" dstbuf="i" dstoff="{(sendFrom0//instances)*ch}" cnt="{sendFrom0//instances}" depid="-1" deps="-1" hasdep="0"/>')
            print(f'    </tb>')
            tbindex+=1
            for j in range(1,ngpuspernode):
                print(f'    <tb id="{tbindex}" send="-1" recv="{j+ngpuspernode}" chan="{ch}">')
                print(f'      <step s="0" type="r" srcbuf="s" srcoff="{ch}" dstbuf="i" dstoff="{offset+j*instances+ch}" cnt="1" depid="-1" deps="-1" hasdep="0"/>')
                print(f'    </tb>')
                tbindex+=1

        if i < ngpuspernode and i > 0:
            print(f'    <tb id="{tbindex}" send="{i+ngpuspernode}" recv="0" chan="{ch}">')
            print(f'      <step s="0" type="rcs" srcbuf="s" srcoff="{ch}" dstbuf="s" dstoff="{ch}" cnt="1" depid="-1" deps="-1" hasdep="0"/>')
            print(f'    </tb>')
            tbindex+=1

        if i > ngpuspernode:
            print(f'    <tb id="{tbindex}" send="{ngpuspernode}" recv="{i-ngpuspernode}" chan="{ch}">')
            print(f'      <step s="0" type="rcs" srcbuf="s" srcoff="{ch}" dstbuf="s" dstoff="{ch}" cnt="1" depid="-1" deps="-1" hasdep="0"/>')
            print(f'    </tb>')
            tbindex+=1
    print('  </gpu>')
print('</algo>')

