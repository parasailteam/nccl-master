nnodes = 1
ngpuspernode = 8
instances = 1
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
        print(f'  <gpu id="{node*ngpuspernode+g}" i_chunks="{nchunksperloop}" o_chunks="{nchunksperloop}" s_chunks="{instances*2*ngpuspernode**2}">')
        for ch in range(instances):
            # print('{{tbindex}}:{{0}} s[{{ch*ngpuspernode**2}}] send {{crossnodenghr}}.s[{{instances*ngpuspernode**2+ch*ngpuspernode**2}}] {{ch}}')
            # print('{{instances*(2+2*g)+ch}}:{{ngpuspernode}} -> {{tbindex}}{{0}}')
            print(f'    <tb id="{tbindex}" send="{crossnodenghr}" recv="-1" chan="{ch}">')
            print(f'      <step s="0" type="s" srcbuf="s" srcoff="{ch*ngpuspernode**2}" dstbuf="s" dstoff="{instances*ngpuspernode**2+ch*ngpuspernode**2}" cnt="{ngpuspernode**2}" depid="{instances*(2+2*g)+ch}" deps="{ngpuspernode}" hasdep="0"/>')
            print(f'    </tb>')
            tbindex+=1
        for ch in range(instances):
            print(f'    <tb id="{tbindex}" send="-1" recv="{crossnodenghr}" chan="{ch}">')
            print(f'      <step s="0" type="r" srcbuf="s" srcoff="{ch*ngpuspernode**2}" dstbuf="s" dstoff="{instances*ngpuspernode**2+ch*ngpuspernode**2}" cnt="{ngpuspernode**2}" depid="-1" deps="-1" hasdep="1"/>')
            print(f'    </tb>')
            tbindex+=1
        for withinnodenghr  in range(ngpuspernode):
            withinNghrNode, withinNghrG, withinCrossNodeNghr = CrossNodeNghr(node, withinnodenghr)
            if withinnodenghr == g:
                for ch in range(instances):
                    step = 0
                    print(f'    <tb id="{tbindex}" send="-1" recv="-1" chan="0">')
                    print(f'      <step s="{step}" type="cpy" srcbuf="i" srcoff="{instances*nghrNode*ngpuspernode+ch*ngpuspernode}" dstbuf="s" dstoff="{instances*g*ngpuspernode+ch*ngpuspernode}" cnt="{ngpuspernode}" depid="-1" deps="-1" hasdep="{1}"/>')
                    step += 1
                    for j in range(ch*(ngpuspernode//instances), (ch+1)*(ngpuspernode//instances)):
                        for k in range(instances):
                            print(f'      <step s="{step}" type="nop" srcbuf="i" srcoff="0" dstbuf="o" dstoff="0" cnt="0" depid="{(instances*(2*j+2+1)+k) if j < g else (instances*(2*j+2)+k)}" deps="{0}" hasdep="{1 if step == 1+ngpuspernode-1 else 0}"/>')
                            step += 1
                    print(f'      <step s="{step}" type="cpy" srcbuf="i" srcoff="{instances*(node*ngpuspernode+g)+ch}" dstbuf="o" dstoff="{instances*(node*ngpuspernode+g)+ch}" cnt="{1}" depid="-1" deps="-1" hasdep="0"/>')
                    step += 1
                    for j in range(ngpuspernode):
                        print(f'      <step s="{step}" type="cpy" srcbuf="s" srcoff="{instances*(ngpuspernode**2+j*ngpuspernode+g)+ch}" dstbuf="o" dstoff="{instances*(nghrNode*ngpuspernode+j)+ch}" cnt="{1}" depid="{instances+(instances*(j*ngpuspernode+g)+ch)//((instances*ngpuspernode**2)//instances)}" deps="{0}" hasdep="0"/>')
                        step += 1
                    print(f'    </tb>')
                    tbindex+=1
            else:
                for ch in range(instances):
                    print(f'    <tb id="{tbindex}" send="{node*ngpuspernode+withinnodenghr}" recv="-1" chan="{ch}">')
                    print(f'      <step s="0" type="s" srcbuf="i" srcoff="{instances*withinNghrNode*ngpuspernode+ch*ngpuspernode}" dstbuf="s" dstoff="{instances*g*ngpuspernode+ch*ngpuspernode}" cnt="{ngpuspernode}" depid="-1" deps="-1" hasdep="0"/>')
                    print(f'      <step s="1" type="s" srcbuf="i" srcoff="{instances*(node*ngpuspernode+withinnodenghr)+ch}" dstbuf="o" dstoff="{instances*(node*ngpuspernode+g)+ch}" cnt="{1}" depid="-1" deps="-1" hasdep="0"/>')
                    step = 2
                    for j in range(ngpuspernode):
                        print(f'      <step s="{step}" type="s" srcbuf="s" srcoff="{instances*(ngpuspernode**2+j*ngpuspernode+withinnodenghr)+ch}" dstbuf="o" dstoff="{instances*(nghrNode*ngpuspernode+j)+ch}" cnt="{1}" depid="{ instances+(instances*(j*ngpuspernode+withinnodenghr)+ch)//((instances*ngpuspernode**2)//instances)}" deps="{0}" hasdep="0"/>')
                        step += 1
                    print(f'    </tb>')
                    tbindex+=1
                for ch in range(instances):
                    print(f'    <tb id="{tbindex}" send="-1" recv="{node*ngpuspernode+withinnodenghr}" chan="{ch}">')
                    print(f'      <step s="0" type="r" srcbuf="i" srcoff="{instances*nghrNode*ngpuspernode+ch*ngpuspernode}" dstbuf="s" dstoff="{instances*withinnodenghr*ngpuspernode+ch*ngpuspernode}" cnt="{ngpuspernode}" depid="-1" deps="-1" hasdep="1"/>')
                    print(f'      <step s="1" type="r" srcbuf="i" srcoff="{instances*(node*ngpuspernode+g)+ch}" dstbuf="o" dstoff="{instances*(node*ngpuspernode+withinnodenghr)+ch}" cnt="{1}" depid="-1" deps="-1" hasdep="0"/>')
                    step = 2
                    for j in range(ngpuspernode):
                        print(f'      <step s="{step}" type="r" srcbuf="s" srcoff="{instances*(ngpuspernode**2+j*ngpuspernode+g)+ch}" dstbuf="o" dstoff="{instances*(withinNghrNode*ngpuspernode+j)+ch}" cnt="{1}" depid="-1" deps="-1" hasdep="0"/>')
                        step += 1
                    print(f'    </tb>')
                    tbindex+=1
        print('  </gpu>')
print('</algo>')
