nchunksperloop = 8
instances = 1
ngpus = 8
print('<algo name="test" nchunksperloop="{}" nchannels="{}" proto="LL" ngpus="{}">'.format(nchunksperloop, instances, ngpus))

for i in range(ngpus):
    tbindex = 0
    print('  <gpu id="{}" i_chunks="{}" o_chunks="{}" s_chunks="{}">'.format(i, nchunksperloop, nchunksperloop, nchunksperloop))
    for j in range(ngpus):
        if i == j:
            continue
            # for ch in range(instances):
            #     print('    <tb id="{}" send="-1" recv="-1" chan="0">'.format(tbindex))
            #     print('      <step s="0" type="cpy" srcbuf="i" srcoff="{}" dstbuf="o" dstoff="{}" cnt="1" depid="-1" deps="-1" hasdep="0"/>'.format(0,0))
            #     print('    </tb>')
            #     tbindex+=1
        else:
            for ch in range(instances):
                print('    <tb id="{}" send="{}" recv="{}" chan="{}">'.format(tbindex, j, j, ch))
                print('      <step s="0" type="s" srcbuf="i" srcoff="{}" dstbuf="o" dstoff="{}" cnt="1" depid="-1" deps="-1" hasdep="0"/>'.format(0,0))
                print('      <step s="1" type="r" srcbuf="i" srcoff="{}" dstbuf="o" dstoff="{}" cnt="1" depid="-1" deps="-1" hasdep="0"/>'.format(0,0))
                print('    </tb>')
                tbindex+=1
    print('  </gpu>')
print('</algo>')
