nnodes = 9
ngpuspernode = 8
instances = 2
nchunksperloop = nnodes*ngpuspernode*instances
print('<algo name="test" nchunksperloop="{}" nchannels="{}" proto="Simple">'.format(nchunksperloop, instances))

def CrossNodeNghr(node, g):
    nghrNode = g if node > g else g+1
    nghrG = node if nghrNode > node else node-1
    return nghrNode, nghrG

def Rank(node, g):
    return node * ngpuspernode + g

for node in range(nnodes):
    for g in range(ngpuspernode):
        tbindex = 0
        rank = Rank(node, g)
        nghrNode, nghrG = CrossNodeNghr(node,g)
        remoteRank = Rank(nghrNode, nghrG)

        print(f'  <gpu id="{rank}" i_chunks="{nchunksperloop}" o_chunks="{nchunksperloop}" s_chunks="{instances*2*ngpuspernode**2}">')
        for ch in range(instances):
            # IB_SEND: depends on GATHER_RECV 
            print(f'    <tb id="{tbindex}" send="{remoteRank}" recv="-1" chan="{ch}">')
            print(f'      <step s="0" type="s" srcbuf="s" srcoff="{ch*ngpuspernode**2}" dstbuf="s" dstoff="{instances*ngpuspernode**2+ch*ngpuspernode**2}" cnt="{ngpuspernode**2}" depid="{instances*(2+2*g)+ch}" deps="{ngpuspernode}" hasdep="0"/>')
            print(f'    </tb>')
            tbindex+=1
        for ch in range(instances):
            # IB_Recv: matching to IB_SEND 
            print(f'    <tb id="{tbindex}" send="-1" recv="{remoteRank}" chan="{ch}">')
            print(f'      <step s="0" type="r" srcbuf="s" srcoff="{ch*ngpuspernode**2}" dstbuf="s" dstoff="{instances*ngpuspernode**2+ch*ngpuspernode**2}" cnt="{ngpuspernode**2}" depid="-1" deps="-1" hasdep="1"/>')
            print(f'    </tb>')
            tbindex+=1
        for g2  in range(ngpuspernode):
            nghrNode2, nghrG2 = CrossNodeNghr(node, g2)
            rank2 = Rank(node, g2)
            if g2 == g:
                for ch in range(instances):
                    step = 0
                    print(f'    <tb id="{tbindex}" send="-1" recv="-1" chan="0">')
                    # COPY: copy g's data to be sent to nghrNode
                    print(f'      <step s="{step}" type="cpy" srcbuf="i" srcoff="{instances*nghrNode*ngpuspernode+ch*ngpuspernode}" dstbuf="s" dstoff="{instances*g*ngpuspernode+ch*ngpuspernode}" cnt="{ngpuspernode}" depid="-1" deps="-1" hasdep="{1}"/>')
                    step += 1
                    for j in range(ch*(ngpuspernode//instances), (ch+1)*(ngpuspernode//instances)):
                        for k in range(instances):
                            # satisfies the dependence on GATHER
                            print(f'      <step s="{step}" type="nop" srcbuf="i" srcoff="0" dstbuf="o" dstoff="0" cnt="0" depid="{(instances*(2*j+2+1)+k) if j < g else (instances*(2*j+2)+k)}" deps="{0}" hasdep="{1 if step == 1+ngpuspernode-1 else 0}"/>')
                            step += 1
                    # copy from input buffer to output buffer for local chunk
                    print(f'      <step s="{step}" type="cpy" srcbuf="i" srcoff="{instances*rank+ch}" dstbuf="o" dstoff="{instances*rank+ch}" cnt="{1}" depid="-1" deps="-1" hasdep="0"/>')
                    step += 1
                    for j in range(ngpuspernode):
                        # copy data received from remote neighbor to output buffer
                        print(f'      <step s="{step}" type="cpy" srcbuf="s" srcoff="{instances*(ngpuspernode**2+j*ngpuspernode+g)+ch}" dstbuf="o" dstoff="{instances*Rank(nghrNode,j)+ch}" cnt="{1}" depid="{instances+(instances*(j*ngpuspernode+g)+ch)//((instances*ngpuspernode**2)//instances)}" deps="{0}" hasdep="0"/>')
                        step += 1
                    print(f'    </tb>')
                    tbindex+=1
            else:
                for ch in range(instances):
                    print(f'    <tb id="{tbindex}" send="{rank2}" recv="-1" chan="{ch}">')
                    # GATHER: send all data that needs to go to nghrNode2 by sending data from g to g2
                    print(f'      <step s="0" type="s" srcbuf="i" srcoff="{instances*nghrNode2*ngpuspernode+ch*ngpuspernode}" dstbuf="s" dstoff="{instances*g*ngpuspernode+ch*ngpuspernode}" cnt="{ngpuspernode}" depid="-1" deps="-1" hasdep="0"/>')
                    # LOCALTRANS: send the chunk that needs to go to g2 from g
                    print(f'      <step s="1" type="s" srcbuf="i" srcoff="{instances*rank2+ch}" dstbuf="o" dstoff="{instances*rank+ch}" cnt="{1}" depid="-1" deps="-1" hasdep="0"/>')
                    step = 2
                    # SCATTER: scatter the data that I received from nghrNode2:  Depends on IB_Recv
                    for j in range(ngpuspernode):
                        print(f'      <step s="{step}" type="s" srcbuf="s" srcoff="{instances*(ngpuspernode**2+j*ngpuspernode+g2)+ch}" dstbuf="o" dstoff="{instances*Rank(nghrNode,j)+ch}" cnt="{1}" depid="{ instances+(instances*(j*ngpuspernode+g2)+ch)//((instances*ngpuspernode**2)//instances)}" deps="{0}" hasdep="0"/>')
                        step += 1
                    print(f'    </tb>')
                    tbindex+=1
                for ch in range(instances):
                    print(f'    <tb id="{tbindex}" send="-1" recv="{rank2}" chan="{ch}">')
                    # GATHER_RECV: matching to GATHER
                    print(f'      <step s="0" type="r" srcbuf="i" srcoff="{instances*nghrNode*ngpuspernode+ch*ngpuspernode}" dstbuf="s" dstoff="{instances*g2*ngpuspernode+ch*ngpuspernode}" cnt="{ngpuspernode}" depid="-1" deps="-1" hasdep="1"/>')
                    # LOCALTRANS_RECV: matching to LOCALTRANS
                    print(f'      <step s="1" type="r" srcbuf="i" srcoff="{instances*rank+ch}" dstbuf="o" dstoff="{instances*rank2+ch}" cnt="{1}" depid="-1" deps="-1" hasdep="0"/>')
                    step = 2

                    for j in range(ngpuspernode):
                        # SCATTER_RECV: matching to SCATTER
                        print(f'      <step s="{step}" type="r" srcbuf="s" srcoff="{instances*(ngpuspernode**2+j*ngpuspernode+g)+ch}" dstbuf="o" dstoff="{instances*Rank(nghrNode2,j)+ch}" cnt="{1}" depid="-1" deps="-1" hasdep="0"/>')
                        step += 1
                    print(f'    </tb>')
                    tbindex+=1
        print('  </gpu>')
print('</algo>')
