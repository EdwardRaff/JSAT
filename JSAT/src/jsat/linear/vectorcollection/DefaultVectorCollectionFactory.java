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
            return new VectorArray<V>(distanceMetric, source);
        if(distanceMetric.isSymmetric() && distanceMetric.isIndiscemible() && distanceMetric.isSubadditive())
            return new VPTreeMV<V>(source, distanceMetric);
        return new VectorArray<V>(distanceMetric, source);
    }

    @Override
    public VectorCollection<V> getVectorCollection(List<V> source, DistanceMetric distanceMetric, ExecutorService threadpool)
    {
        if(source.size() < VEC_ARRAY_CUT_OFF)
            return new VectorArray<V>(distanceMetric, source);
        if(distanceMetric.isSymmetric() && distanceMetric.isIndiscemible() && distanceMetric.isSubadditive())
            return new VPTreeMV<V>(source, distanceMetric, VPTree.VPSelection.Random, RandomUtil.getRandom(), 1, 1, threadpool);
        return new VectorArray<V>(distanceMetric, source);
    }

    @Override
    public DefaultVectorCollectionFactory<V> clone()
    {
        return new DefaultVectorCollectionFactory<V>();
    }
    
}
