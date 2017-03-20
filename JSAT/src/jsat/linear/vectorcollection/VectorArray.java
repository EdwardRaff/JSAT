
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
import jsat.utils.DoubleList;
import jsat.utils.ProbailityMatch;

/**
 * This is the naive implementation of a Vector collection. Construction time is 
 * O(n) only to clone the n elements, and all queries are O(n)
 * <br><br>
 * Removing elements from the vector array will result in the destruction of any
 * {@link DistanceMetric#getAccelerationCache(java.util.List) acceleration cache}
 * 
 * @author Edward Raff
 */
public class VectorArray<V extends Vec> extends ArrayList<V> implements IncrementalCollection<V>
{
    private static final long serialVersionUID = 5365949686370986234L;
    private DistanceMetric distanceMetric;
    private List<Double> distCache;

    public VectorArray(DistanceMetric distanceMetric, int initialCapacity)
    {
        super(initialCapacity);
        this.distanceMetric = distanceMetric;
        if(distanceMetric.supportsAcceleration())
            distCache = new DoubleList(initialCapacity);
    }

    public VectorArray(DistanceMetric distanceMetric, Collection<? extends V> c)
    {
        super(c);
        this.distanceMetric = distanceMetric;
        if(distanceMetric.supportsAcceleration())
            distCache = distanceMetric.getAccelerationCache(this);
    }

    public VectorArray(DistanceMetric distanceMetric)
    {
        super();
        this.distanceMetric = distanceMetric;
        if(distanceMetric.supportsAcceleration())
            distCache = new DoubleList();
    }

    public DistanceMetric getDistanceMetric()
    {
        return distanceMetric;
    }

    public void setDistanceMetric(DistanceMetric distanceMetric)
    {
        this.distanceMetric = distanceMetric;
        if(distanceMetric.supportsAcceleration())
            this.distCache = distanceMetric.getAccelerationCache(this);
        else
            this.distCache = null;
    }
    
    @Override
    public void insert(V x)
    {
        add(x);
    }

    @Override
    public boolean add(V e)
    {
        boolean toRet = super.add(e);
        if(distCache != null)
            this.distCache.addAll(distanceMetric.getQueryInfo(e));
        return toRet;
    }

    @Override
    public boolean addAll(Collection<? extends V> c)
    {
        boolean toRet = super.addAll(c);
        if(this.distCache != null)
            for(V v : c)
                this.distCache.addAll(this.distanceMetric.getQueryInfo(v));
        return toRet;
    }

    @Override
    public V remove(int index)
    {
        distCache = null;
        return super.remove(index); //To change body of generated methods, choose Tools | Templates.
    }
    

    @Override
    public List<? extends VecPaired<V, Double>> search(Vec query, double range)
    {
        List<VecPairedComparable<V, Double>> list = new ArrayList<VecPairedComparable<V, Double>>();
        
        List<Double> qi = distanceMetric.getQueryInfo(query);
        
        for(int i = 0; i < size(); i++)
        {
            double distance = distanceMetric.dist(i, query, qi, this, distCache);
            if(distance <= range)
                list.add(new VecPairedComparable<V, Double>(get(i), distance));
        }
        Collections.sort(list);
        return list;
    }

    @Override
    public List<? extends VecPaired<V, Double>> search(Vec query, int neighbors)
    {
        BoundedSortedList<ProbailityMatch<V>> knns = new BoundedSortedList<ProbailityMatch<V>>(neighbors);
        
        List<Double> qi = distanceMetric.getQueryInfo(query);
        
        for(int i = 0; i < size(); i++)
        {
            double distance = distanceMetric.dist(i, query, qi, this, distCache);
            knns.add(new ProbailityMatch<V>(distance, get(i)));
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
        VectorArray<V> clone = new VectorArray<V>(distanceMetric, this);
        
        return clone;
    }

    public static class VectorArrayFactory<V extends Vec> implements VectorCollectionFactory<V>
    {
        private static final long serialVersionUID = -7470849503958877157L;

        @Override
        public VectorCollection<V> getVectorCollection(List<V> source, DistanceMetric distanceMetric)
        {
            return new VectorArray<V>(distanceMetric, source);
        }

        @Override
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
