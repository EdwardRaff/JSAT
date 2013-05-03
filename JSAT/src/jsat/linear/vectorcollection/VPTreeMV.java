
package jsat.linear.vectorcollection;

import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.vectorcollection.VPTree.VPSelection;
import jsat.math.OnLineStatistics;
import jsat.utils.ProbailityMatch;

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

    public VPTreeMV(List<V> list, DistanceMetric dm, VPSelection vpSelection, Random rand, int sampleSize, int searchIterations, ExecutorService threadpool)
    {
        super(list, dm, vpSelection, rand, sampleSize, searchIterations, threadpool);
    }

    public VPTreeMV(List<V> list, DistanceMetric dm, VPSelection vpSelection, Random rand, int sampleSize, int searchIterations)
    {
        super(list, dm, vpSelection, rand, sampleSize, searchIterations);
    }

    public VPTreeMV(List<V> list, DistanceMetric dm, VPSelection vpSelection)
    {
        super(list, dm, vpSelection);
    }

    public VPTreeMV(List<V> list, DistanceMetric dm)
    {
        super(list, dm);
    }

    @Override
    protected int splitListIndex(List<ProbailityMatch<V>> S)
    {
        int splitIndex = S.size()/2;
        int maxLeafSize = getMaxLeafSize();
        
        if(S.size() >= maxLeafSize*4)
        {
            OnLineStatistics rightV = new OnLineStatistics();
            OnLineStatistics leftV = new OnLineStatistics();
            for(int i = 0; i < maxLeafSize; i++)
                leftV.add(S.get(i).getProbability());
            for(int i = maxLeafSize; i < S.size(); i++)
                rightV.add(S.get(i).getProbability());
            splitIndex = maxLeafSize;
            double bestVar = leftV.getVarance()*maxLeafSize+rightV.getVarance()*(S.size()-maxLeafSize);
            for(int i = maxLeafSize+1; i < S.size()-maxLeafSize; i++)
            {
                double tmp = S.get(i).getProbability();
                leftV.add(tmp);
                rightV.remove(tmp, 1.0);
                double testVar = leftV.getVarance()*i + rightV.getVarance()*(S.size()-i);
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
        private VPSelection vpSelectionMethod;

        public VPTreeMVFactory(VPSelection vpSelectionMethod)
        {
            this.vpSelectionMethod = vpSelectionMethod;
        }

        public VPTreeMVFactory()
        {
            this(VPSelection.Random);
        }
        
        @Override
        public VectorCollection<V> getVectorCollection(List<V> source, DistanceMetric distanceMetric)
        {
            return new VPTreeMV<V>(source, distanceMetric, vpSelectionMethod);
        }

        @Override
        public VectorCollection<V> getVectorCollection(List<V> source, DistanceMetric distanceMetric, ExecutorService threadpool)
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
