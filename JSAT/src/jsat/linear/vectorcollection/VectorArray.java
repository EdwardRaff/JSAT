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
 * O(n) only to clone the n elements, and all queries are O(n) <br>
 * <br>
 * Removing elements from the vector array will result in the destruction of any
 * {@link DistanceMetric#getAccelerationCache(java.util.List) acceleration
 * cache}
 *
 * @author Edward Raff
 */
public class VectorArray<V extends Vec> extends ArrayList<V>implements VectorCollection<V> {

  public static class VectorArrayFactory<V extends Vec> implements VectorCollectionFactory<V> {

    /**
     *
     */
    private static final long serialVersionUID = -7470849503958877157L;

    @Override
    public VectorArrayFactory<V> clone() {
      return new VectorArrayFactory<V>();
    }

    @Override
    public VectorCollection<V> getVectorCollection(final List<V> source, final DistanceMetric distanceMetric) {
      return new VectorArray<V>(distanceMetric, source);
    }

    @Override
    public VectorCollection<V> getVectorCollection(final List<V> source, final DistanceMetric distanceMetric,
        final ExecutorService threadpool) {
      return getVectorCollection(source, distanceMetric);
    }
  }

  private static final long serialVersionUID = 5365949686370986234L;
  private DistanceMetric distanceMetric;

  private List<Double> distCache;

  public VectorArray(final DistanceMetric distanceMetric) {
    super();
    this.distanceMetric = distanceMetric;
    if (distanceMetric.supportsAcceleration()) {
      distCache = new DoubleList();
    }
  }

  public VectorArray(final DistanceMetric distanceMetric, final Collection<? extends V> c) {
    super(c);
    this.distanceMetric = distanceMetric;
    if (distanceMetric.supportsAcceleration()) {
      distCache = distanceMetric.getAccelerationCache(this);
    }
  }

  public VectorArray(final DistanceMetric distanceMetric, final int initialCapacity) {
    super(initialCapacity);
    this.distanceMetric = distanceMetric;
    if (distanceMetric.supportsAcceleration()) {
      distCache = new DoubleList(initialCapacity);
    }
  }

  @Override
  public boolean add(final V e) {
    final boolean toRet = super.add(e);
    if (distCache != null) {
      this.distCache.addAll(distanceMetric.getQueryInfo(e));
    }
    return toRet;
  }

  @Override
  public boolean addAll(final Collection<? extends V> c) {
    final boolean toRet = super.addAll(c);
    if (this.distCache != null) {
      for (final V v : c) {
        this.distCache.addAll(this.distanceMetric.getQueryInfo(v));
      }
    }
    return toRet;
  }

  @Override
  public VectorArray<V> clone() {
    final VectorArray<V> clone = new VectorArray<V>(distanceMetric, this);

    return clone;
  }

  public DistanceMetric getDistanceMetric() {
    return distanceMetric;
  }

  @Override
  public V remove(final int index) {
    distCache = null;
    return super.remove(index); // To change body of generated methods, choose
                                // Tools | Templates.
  }

  @Override
  public List<? extends VecPaired<V, Double>> search(final Vec query, final double range) {
    final List<VecPairedComparable<V, Double>> list = new ArrayList<VecPairedComparable<V, Double>>();

    final List<Double> qi = distanceMetric.getQueryInfo(query);

    for (int i = 0; i < size(); i++) {
      final double distance = distanceMetric.dist(i, query, qi, this, distCache);
      if (distance <= range) {
        list.add(new VecPairedComparable<V, Double>(get(i), distance));
      }
    }
    Collections.sort(list);
    return list;
  }

  @Override
  public List<? extends VecPaired<V, Double>> search(final Vec query, final int neighbors) {
    final BoundedSortedList<ProbailityMatch<V>> knns = new BoundedSortedList<ProbailityMatch<V>>(neighbors);

    final List<Double> qi = distanceMetric.getQueryInfo(query);

    for (int i = 0; i < size(); i++) {
      final double distance = distanceMetric.dist(i, query, qi, this, distCache);
      knns.add(new ProbailityMatch<V>(distance, get(i)));
    }

    final List<VecPaired<V, Double>> knnsList = new ArrayList<VecPaired<V, Double>>(knns.size());
    for (int i = 0; i < knns.size(); i++) {
      final ProbailityMatch<V> pm = knns.get(i);
      knnsList.add(new VecPaired<V, Double>(pm.getMatch(), pm.getProbability()));
    }

    return knnsList;

  }

  public void setDistanceMetric(final DistanceMetric distanceMetric) {
    this.distanceMetric = distanceMetric;
    if (distanceMetric.supportsAcceleration()) {
      this.distCache = distanceMetric.getAccelerationCache(this);
    } else {
      this.distCache = null;
    }
  }

}
