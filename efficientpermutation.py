import numpy
import sys

# From http://code.activestate.com/recipes/113799-bit-field-manipulation/
class bf(object):
    def __init__(self,value=0):
        self._d = value

    def __getitem__(self, index):
        return (self._d >> index) & 1 

    def __setitem__(self,index,value):
        value    = (value&1L)<<index
        mask     = (1L)<<index
        self._d  = (self._d & ~mask) | value

    def __getslice__(self, start, end):
        mask = 2L**(end - start) -1
        return (self._d >> start) & mask

    def __setslice__(self, start, end, value):
        mask = 2L**(end - start) -1
        value = (value & mask) << start
        mask = mask << start
        self._d = (self._d & ~mask) | value
        return (self._d >> start) & mask

    def __int__(self):
        return self._d

class EfficientPermutation:
    # Generate a uniform permutation of {0,...n-1}, for n up to 2**31,
    # sequentially in a fast and memory efficient manner
    def __init__( self, n ):
        n = long(n)
        self.n = n
        self.depth = int(numpy.ceil(numpy.log2(n)))+1
        
        # A compact binary representation of a tree
        self.bfs = [ bf( 2**n - 1) ]
        for i in range(2,self.depth+1):
            bits_per_counter = i
            counters = n/(2**(i-1))+1
            r = 2**bits_per_counter
            self.bfs.append( bf( r/2 * (r**(counters-1) -1)/(r-1) ) ) # All counters except last, which will all be set to the maximum value
            self.bfs[-1][ bits_per_counter * (counters - 1) : bits_per_counter * counters ] = n%(2**(i-1)) # Last counter, which may have a lower value
    
    def left_to_iterate( self ):
        return self.bfs[-1]._d
    
    def random_set_bit( self ):
        # Return the index of a random bit that is set in O(log n) time
        offset = 0
        val = 0
        for i in range(self.depth-2,-1,-1):
            bits_per_counter = i+1            
            countbranch1 = self.bfs[i][offset*bits_per_counter:(offset+1)*bits_per_counter]
            countbranch2 = self.bfs[i][(offset+1)*bits_per_counter:(offset+2)*bits_per_counter]            
            if self.flip_coin( countbranch1, countbranch2 ):
                # Selected branch 1
                offset = offset*2                
            else:
                # Selected branch 2
                offset = offset*2+2
                val = val + 2**i
        return val
    
    def unset_bit( self, bit ):
        assert( self.yet_to_appear(bit) )
        offset = long(bit)
        for i in range(0, self.depth):
            bits_per_counter = i+1
            self.bfs[i][bits_per_counter*offset:bits_per_counter*(offset+1)] = self.bfs[i][bits_per_counter*offset:bits_per_counter*(offset+1)] -1
            offset = offset >> 1    
    
    def yet_to_appear( self, bit ):
        return self.bfs[0][long(bit)]
    
    def flip_coin( self, a, b ):
        return numpy.random.uniform()<float(a)/float(a+b)

    def next( self ):
        #print "1"
        assert( self.left_to_iterate()>0 )     
        #print "2"
        ret = self.random_set_bit()
        #print "3"
        self.unset_bit( ret )
        return ret
        
        
if __name__ == "__main__":    
    #numpy.random.seed(0)
    #e = EfficientPermutation( long(10) )
    #e.unset_bit(4)
    #for i in range(20):
        #print e.random_set_bit()
    
    #exit(0)
    
    print "Testing permuting very large set"
    e = EfficientPermutation( 22*1000*1000*10 )
    print numpy.log2(22*1000*1000*10)
    print "running"
    for i in range(512):
        print e.next(),
        sys.stdout.flush()
    print

    print "testing next (output should be close to 1)"
    count = {}    
    for i in range(100000):
        e = EfficientPermutation( 4 )
        perm = []
        for j in range(4):
            perm.append(e.next())        
        try:
            count[tuple(perm)]+=1
        except:
            count[tuple(perm)]=1
    for perm in count.keys():
        print perm,count[perm]/100000.0*(4*3*2*1)
    
    print "testing set random_set_bit (output should be close to zero"
    for n in numpy.hstack( (numpy.random.random_integers(10, 100, 5),[2**4, 2**4-1, 2**4+1])):        
        e = EfficientPermutation( long(n) )
        remove = numpy.random.random_integers(0,1,n)
        for b in numpy.where( remove )[0]:
            e.unset_bit(b)
        
        count = numpy.zeros( n, dtype=numpy.int32 )
        for i in range(10000*n):
            a = e.random_set_bit()
            count[a]+=1
        print count/float(10000*n)*sum(remove==0) - (remove==0)
