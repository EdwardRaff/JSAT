
package jsat.linear.vectorcollection;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.utils.BoundedSortedList;
import jsat.utils.ProbailityMatch;

/**
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

    public List<VecPaired<Double, V>> search(Vec query, double range)
    {
        List<VecPaired<Double,V>> list = new ArrayList<VecPaired<Double,V>>();
        
        for(V v : this)
        {
            double distance = distanceMetric.dist(query, VecPaired.extractTrueVec(v));
            if(distance <= range)
                list.add(new VecPaired<Double, V>(v, distance));
        }
        
        return list;
    }

    public List<VecPaired<Double, V>> search(Vec query, int neighbors)
    {
        BoundedSortedList<ProbailityMatch<V>> knns = new BoundedSortedList<ProbailityMatch<V>>(neighbors);
        
        for(V v : this)
        {
            double distance = distanceMetric.dist(query, VecPaired.extractTrueVec(v));
            knns.add(new ProbailityMatch<V>(distance, v));
        }
        
        List<VecPaired<Double,V>> knnsList = new ArrayList<VecPaired<Double,V>>(knns.size());
        for(int i = 0; i < knns.size(); i++)
        {
            ProbailityMatch<V> pm = knns.get(i);
            knnsList.add(new VecPaired<Double, V>(pm.getMatch(), pm.getProbability()));
        }
                
        return knnsList;
        
    }
    
    public static class VectorArrayFactory<V extends Vec> implements VectorCollectionFactory<V>
    {
        public VectorCollection<V> getVectorCollection(List<V> source, DistanceMetric distanceMetric)
        {
            return new VectorArray<V>(distanceMetric, source);
        }
    }
    
}
