
package jsat.linear.vectorcollection;

import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.math.OnLineStatistics;
import jsat.utils.Pair;

/**
 * The VPTreeMV is an extension of the VPTree, the MV meaning "of Minimum Variance". This extension 
 * alters the splitting method of nodes, giving up the O(log n) bound on query time. This is done 
 * to reduce the variance in the distance to the parent node of each split, which can result in 
 * lopsided distributions of values for each split. At the same time, this lopsidedness may better 
 * reflect the locality of points in the data set. This can result in a decrease in query time 
 * for some data sets, with minimal impact on construction time. 
 * 
 * @author Edward Raff
 */
public class VPTreeMV<V extends Vec> extends VPTree<V>
{


	private static final long serialVersionUID = 6668184445206226077L;

	public VPTreeMV(final List<V> list, final DistanceMetric dm, final VPSelection vpSelection, final Random rand, final int sampleSize, final int searchIterations, final ExecutorService threadpool)
    {
        super(list, dm, vpSelection, rand, sampleSize, searchIterations, threadpool);
    }

    public VPTreeMV(final List<V> list, final DistanceMetric dm, final VPSelection vpSelection, final Random rand, final int sampleSize, final int searchIterations)
    {
        super(list, dm, vpSelection, rand, sampleSize, searchIterations);
    }

    public VPTreeMV(final List<V> list, final DistanceMetric dm, final VPSelection vpSelection)
    {
        super(list, dm, vpSelection);
    }

    public VPTreeMV(final List<V> list, final DistanceMetric dm)
    {
        super(list, dm);
    }

    @Override
    protected int splitListIndex(final List<Pair<Double, Integer>> S)
    {
        int splitIndex = S.size()/2;
        final int maxLeafSize = getMaxLeafSize();
        
        if(S.size() >= maxLeafSize*4)
        {
            final OnLineStatistics rightV = new OnLineStatistics();
            final OnLineStatistics leftV = new OnLineStatistics();
            for(int i = 0; i < maxLeafSize; i++) {
              leftV.add(S.get(i).getFirstItem());
            }
            for(int i = maxLeafSize; i < S.size(); i++) {
              rightV.add(S.get(i).getFirstItem());
            }
            splitIndex = maxLeafSize;
            double bestVar = leftV.getVarance()*maxLeafSize+rightV.getVarance()*(S.size()-maxLeafSize);
            for(int i = maxLeafSize+1; i < S.size()-maxLeafSize; i++)
            {
                final double tmp = S.get(i).getFirstItem();
                leftV.add(tmp);
                rightV.remove(tmp, 1.0);
                final double testVar = leftV.getVarance()*i + rightV.getVarance()*(S.size()-i);
                if(testVar < bestVar)
                {
                    splitIndex = i;
                    bestVar = testVar;
                }
            }
        }
        
        return splitIndex;
    }
    
    public static class VPTreeMVFactory<V extends Vec> implements VectorCollectionFactory<V>
    {
        /**
		 * 
		 */
		private static final long serialVersionUID = 4265451324896792148L;
		private VPSelection vpSelectionMethod;

        public VPTreeMVFactory(final VPSelection vpSelectionMethod)
        {
            this.vpSelectionMethod = vpSelectionMethod;
        }

        public VPTreeMVFactory()
        {
            this(VPSelection.Random);
        }
        
        @Override
        public VectorCollection<V> getVectorCollection(final List<V> source, final DistanceMetric distanceMetric)
        {
            return new VPTreeMV<V>(source, distanceMetric, vpSelectionMethod);
        }

        @Override
        public VectorCollection<V> getVectorCollection(final List<V> source, final DistanceMetric distanceMetric, final ExecutorService threadpool)
        {
            return new VPTreeMV<V>(source, distanceMetric, vpSelectionMethod, new Random(10), 80, 40, threadpool);
        }

        @Override
        public VectorCollectionFactory<V> clone()
        {
            return new VPTreeMVFactory<V>(vpSelectionMethod);
        }
    }
}
