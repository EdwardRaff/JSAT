
package jsat.linear.vectorcollection;

import java.util.List;
import java.util.Random;
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

    public VPTreeMV(List<V> list, DistanceMetric dm, VPSelection vpSelection, Random rand, int sampleSize, int searchIterations, boolean parallel)
    {
        super(list, dm, vpSelection, rand, sampleSize, searchIterations, parallel);
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

    public VPTreeMV(List<V> list, DistanceMetric dm, boolean parallel)
    {
        super(list, dm, parallel);
    }

    public VPTreeMV(DistanceMetric dm)
    {
        super(dm);
    }
    
    public VPTreeMV(DistanceMetric dm, VPSelection sampling)
    {
        super(dm, sampling);
    }

    public VPTreeMV()
    {
        super();
    }
    
    public VPTreeMV(VPTreeMV toClone)
    {
        super(toClone);
    }

    @Override
    protected int splitListIndex(List<Pair<Double, Integer>> S)
    {
        int splitIndex = S.size()/2;
        int maxLeafSize = getMaxLeafSize();
        
        if(S.size() >= maxLeafSize*4)
        {
            //Adjust to avoid degenerate cases that create a long string of tiny splits. Most imbalacned slpit can be 1:20
            int minSplitSize = Math.max(maxLeafSize, S.size()/20);
            
            OnLineStatistics rightV = new OnLineStatistics();
            OnLineStatistics leftV = new OnLineStatistics();
            for(int i = 0; i < minSplitSize; i++)
                leftV.add(S.get(i).getFirstItem());
            for(int i = minSplitSize; i < S.size(); i++)
                rightV.add(S.get(i).getFirstItem());
            splitIndex = minSplitSize;
            double bestVar = leftV.getVarance()*minSplitSize+rightV.getVarance()*(S.size()-minSplitSize);
            for(int i = minSplitSize+1; i < S.size()-minSplitSize; i++)
            {
                double tmp = S.get(i).getFirstItem();
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

    @Override
    public VPTreeMV<V> clone()
    {
        return new VPTreeMV<>(this);
    }
}
