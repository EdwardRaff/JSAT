package jsat.linear.vectorcollection;

import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.*;
import jsat.utils.random.RandomUtil;

/**
 * DefaultVectorCollectionFactory is a generic factory that attempts to return a
 * good vector collection for the given input. It may take into account the size
 * of the data set, dimensions, and the distance metric in use to select a 
 * Vector Collection that will have the highest overall performance. 
 * 
 * @author Edward Raff
 */
public class DefaultVectorCollectionFactory<V extends Vec> implements VectorCollectionFactory<V>
{

	private static final long serialVersionUID = -7442543159507721642L;
	private static final int VEC_ARRAY_CUT_OFF = 20; 
    private static final int KD_TREE_CUT_OFF = 14;
    private static final int KD_TREE_PIVOT = 5;
    private static final int BRUTE_FORCE_DIM = 1000;
    @Override
    public VectorCollection<V> getVectorCollection(List<V> source, DistanceMetric distanceMetric)
    {
        if(source.size() < VEC_ARRAY_CUT_OFF)
        {
            return new VectorArray<V>(distanceMetric, source);
        }
        int dimension = source.get(0).length();
        if(dimension >= BRUTE_FORCE_DIM)
        {
            return new VectorArray<V>(distanceMetric, source);
        }
        if(dimension < KD_TREE_CUT_OFF && 
                ( distanceMetric instanceof EuclideanDistance || 
                distanceMetric instanceof ChebyshevDistance || 
                distanceMetric instanceof ManhattanDistance || 
                distanceMetric instanceof MinkowskiDistance))
        {
            KDTree.PivotSelection pivotSelect = dimension <= KD_TREE_PIVOT ? KDTree.PivotSelection.Variance : KDTree.PivotSelection.Incremental;
            KDTree<V> kd = new KDTree<V>(source, distanceMetric, pivotSelect);
            return kd;
        }
        
        return new VPTree<V>(source, distanceMetric, VPTree.VPSelection.Random, RandomUtil.getRandom(), 50, 50);
    }

    @Override
    public VectorCollection<V> getVectorCollection(List<V> source, DistanceMetric distanceMetric, ExecutorService threadpool)
    {
        if(source.size() < VEC_ARRAY_CUT_OFF)
        {
            return new VectorArray<V>(distanceMetric, source);
        }
        int dimension = source.get(0).length();
        if(dimension >= BRUTE_FORCE_DIM)
        {
            return new VectorArray<V>(distanceMetric, source);
        }
        if(dimension < KD_TREE_CUT_OFF && 
                ( distanceMetric instanceof EuclideanDistance || 
                distanceMetric instanceof ChebyshevDistance || 
                distanceMetric instanceof ManhattanDistance || 
                distanceMetric instanceof MinkowskiDistance))
        {
            KDTree.PivotSelection pivotSelect = dimension <= KD_TREE_PIVOT ? KDTree.PivotSelection.Variance : KDTree.PivotSelection.Incremental;
            KDTree<V> kd = new KDTree<V>(source, distanceMetric, pivotSelect, threadpool);
            return kd;
        }
        return new VPTree<V>(source, distanceMetric, VPTree.VPSelection.Random, RandomUtil.getRandom(), 50, 50, threadpool);
    }

    @Override
    public VectorCollectionFactory<V> clone()
    {
        return new DefaultVectorCollectionFactory<V>();
    }
    
}
