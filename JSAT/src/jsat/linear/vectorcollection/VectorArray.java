
package jsat.linear.vectorcollection;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.linear.VecPairedComparable;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.utils.BoundedSortedList;
import jsat.utils.ProbailityMatch;

/**
 * This is the naive implementation of a Vector collection. Construction time is 
 * O(n) only to clone the n elements, and all queries are O(n<sup>2</sup>)
 * 
 * @author Edward Raff
 */
public class VectorArray<V extends Vec> extends ArrayList<V> implements VectorCollection<V>
{
    private DistanceMetric distanceMetric;

    public VectorArray(DistanceMetric distanceMetric, int initialCapacity)
    {
        super(initialCapacity);
        this.distanceMetric = distanceMetric;
    }

    public VectorArray(DistanceMetric distanceMetric, Collection<? extends V> c)
    {
        super(c);
        this.distanceMetric = distanceMetric;
    }

    public VectorArray(DistanceMetric distanceMetric)
    {
        super();
        this.distanceMetric = distanceMetric;
    }

    public DistanceMetric getDistanceMetric()
    {
        return distanceMetric;
    }

    public void setDistanceMetric(DistanceMetric distanceMetric)
    {
        this.distanceMetric = distanceMetric;
    }

    @Override
    public List<? extends VecPaired<V, Double>> search(Vec query, double range)
    {
        List<VecPairedComparable<V, Double>> list = new ArrayList<VecPairedComparable<V, Double>>();
        
        for(V v : this)
        {
            double distance = distanceMetric.dist(query, VecPaired.extractTrueVec(v));
            if(distance <= range)
                list.add(new VecPairedComparable<V, Double>(v, distance));
        }
        Collections.sort(list);
        return list;
    }

    @Override
    public List<? extends VecPaired<V, Double>> search(Vec query, int neighbors)
    {
        BoundedSortedList<ProbailityMatch<V>> knns = new BoundedSortedList<ProbailityMatch<V>>(neighbors);
        
        for(V v : this)
        {
            double distance = distanceMetric.dist(query, VecPaired.extractTrueVec(v));
            knns.add(new ProbailityMatch<V>(distance, v));
        }
        
        List<VecPaired<V, Double>> knnsList = new ArrayList<VecPaired<V, Double>>(knns.size());
        for(int i = 0; i < knns.size(); i++)
        {
            ProbailityMatch<V> pm = knns.get(i);
            knnsList.add(new VecPaired<V, Double>(pm.getMatch(), pm.getProbability()));
        }
                
        return knnsList;
        
    }

    @Override
    public VectorArray<V> clone()
    {
        VectorArray<V> clone = new VectorArray<V>(distanceMetric, size());
        for(V v : this)
            clone.add((V)v.clone());
        return clone;
    }
    
    public static class VectorArrayFactory<V extends Vec> implements VectorCollectionFactory<V>
    {
        public VectorCollection<V> getVectorCollection(List<V> source, DistanceMetric distanceMetric)
        {
            return new VectorArray<V>(distanceMetric, source);
        }

        public VectorCollection<V> getVectorCollection(List<V> source, DistanceMetric distanceMetric, ExecutorService threadpool)
        {
            return getVectorCollection(source, distanceMetric);
        }

        @Override
        public VectorArrayFactory<V> clone()
        {
            return new VectorArrayFactory<V>();
        }
    }
    
}
