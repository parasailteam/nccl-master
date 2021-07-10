nnodes = 9
ngpuspernode = 8
instances = 2
nchunksperloop = nnodes*ngpuspernode*instances
print('<algo name="test" nchunksperloop="{}" nchannels="{}" proto="Simple">'.format(nchunksperloop, instances))

def CrossNodeNghr(node, g):
    nghrNode = g if node > g else g+1
    nghrG = node if nghrNode > node else node-1
    return nghrNode, nghrG, nghrNode * ngpuspernode + nghrG
for node in range(nnodes):
    for g in range(ngpuspernode):
        tbindex = 0
        nghrNode, nghrG, crossnodenghr = CrossNodeNghr(node,g)
        print('  <gpu id="{}" i_chunks="{}" o_chunks="{}" s_chunks="{}">'.format(node*ngpuspernode+g, nchunksperloop, nchunksperloop, instances*2*ngpuspernode**2))
        for ch in range(instances):
            print('    <tb id="{}" send="{}" recv="-1" chan="{}">'.format(tbindex, crossnodenghr, ch))
            print('      <step s="0" type="s" srcbuf="s" srcoff="{}" dstbuf="s" dstoff="{}" cnt="{}" depid="-1" deps="-1" hasdep="0"/>'.format(ch*ngpuspernode**2, instances*ngpuspernode**2+ch*ngpuspernode**2, ngpuspernode**2, instances*(2+2*g)+ch, ngpuspernode-1))
            print('    </tb>')
            tbindex+=1
        for ch in range(instances):
            print('    <tb id="{}" send="-1" recv="{}" chan="{}">'.format(tbindex, crossnodenghr, ch))
            print('      <step s="0" type="r" srcbuf="s" srcoff="{}" dstbuf="s" dstoff="{}" cnt="{}" depid="-1" deps="-1" hasdep="1"/>'.format(ch*ngpuspernode**2, instances*ngpuspernode**2+ch*ngpuspernode**2, ngpuspernode**2))
            print('    </tb>')
            tbindex+=1
        print('  </gpu>')
print('</algo>')
