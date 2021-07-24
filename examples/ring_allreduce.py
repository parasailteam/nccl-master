num_gpus = 16
nchannels = 24
num_chunks = num_gpus * nchannels # 96
num_chunks_ring = num_chunks    # 48
is_overlap = True
def ring_16():
    with open("allreduce_ring_%d_ranks_%d_channel_2D_chunks%s.xml"%(num_gpus, nchannels, "_overlap" if (is_overlap) else ""), "w") as f:
        f.write('<algo name="test_{}_ring8" nchunksperloop="{}" nchannels="{}" ld="1024" chunkld="1024" redop="sum" proto="Simple">\n'.format(num_gpus, num_chunks, nchannels))
        cnt_chunks = num_chunks_ring / nchannels # 8 = num_gpus/2
        # Ring 16
        for g in range(num_gpus):
            f.write('  <gpu id="{}" i_chunks="{}" o_chunks="{}" s_chunks="0">\n'.format(g, num_chunks, num_chunks)) #o_chunks is 1 because this is not inplace
            tbid = -1
            recvpeer =  (g+num_gpus-1)%num_gpus
            sendpeer = (g+1)%num_gpus
            ring = [-1 for g in range(num_gpus)]
            ring[0] = g
            ringidx = 0
            ringval = g
            while ringidx < len(ring):
                ring[ringidx] = ringval
                ringval = (ringval + 1) % num_gpus
                ringidx += 1
            # (recvpeer, sendpeer) = (sendpeer, recvpeer)
            for c in range(nchannels):
                tbid = tbid + 1
                f.write('    <tb id="{}" send="{}" recv="{}" chan="{}">\n'.format(tbid, sendpeer, recvpeer, c))
                # ReduceScatter phase
                for s in range(num_gpus):
                    src_off = ((ring[num_gpus - s - 1]) % num_gpus)*nchannels + c #int(((s+g)*nchannels) % num_chunks_ring + c)
                    depid = tbid if is_overlap else -1
                    deps = s if is_overlap else -1
                    if s == 0:
                        f.write('      <step s="{}" type="s" srcbuf="i" srcoff="{}" dstbuf="o" dstoff="0" cnt="1" depid="{}" deps="{}" hasdep="0"/>\n'.format(s, src_off, depid, deps)) # TODO check
                    elif s == num_gpus-1:
                        f.write('      <step s="{}" type="rrcs" srcbuf="i" srcoff="{}" dstbuf="o" dstoff="{}" cnt="1" depid="{}" deps="{}" hasdep="0"/>\n'.format(s, src_off, src_off, depid, deps))
                    else:
                        f.write('      <step s="{}" type="rrs" srcbuf="i" srcoff="{}" dstbuf="o" dstoff="0" cnt="1" depid="{}" deps="{}" hasdep="0"/>\n'.format(s, src_off, depid, deps))
                
                # AllGather phase
                for s in range(1, num_gpus):
                    dst_off = ((ring[num_gpus - s]) % num_gpus)*nchannels + c #int(((s+g)*nchannels) % num_chunks_ring + c)
                    if s == num_gpus-1:
                        f.write('      <step s="{}" type="r" srcbuf="i" srcoff="0" dstbuf="o" dstoff="{}" cnt="1" depid="-1" deps="-1" hasdep="0"/>\n'.format(s + num_gpus - 1, dst_off))
                    else:
                        f.write('      <step s="{}" type="rcs" srcbuf="i" srcoff="0" dstbuf="o" dstoff="{}" cnt="1" depid="-1" deps="-1" hasdep="0"/>\n'.format(s+ num_gpus - 1, dst_off))
                f.write('    </tb>\n')
            f.write('  </gpu>\n')
        f.write('</algo>\n')
ring_16()