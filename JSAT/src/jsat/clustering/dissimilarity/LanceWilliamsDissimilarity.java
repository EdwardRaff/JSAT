
package jsat.clustering.dissimilarity;

import static java.lang.Math.abs;
import java.util.List;
import java.util.Set;
import jsat.classifiers.DataPoint;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.utils.IntSet;

/**
 * This class provides a base implementation of a Lance Williams (LW) 
 * Dissimilarity measure, which is updatable. All LW measures can be written in the form
 * <br>
 * &alpha;<sub>i</sub> d<sub>ik</sub> + &alpha;<sub>j</sub> d<sub>jk</sub> + 
 * &beta; d<sub>ij</sub> + &gamma; |d<sub>ik</sub> - d<sub>jk</sub>| 
 * <br>
 * The d's represent the distances between points, and the variables: <br> 
 * <ul>
 * <li>&alpha;</li>
 * <li>&beta;</li>
 * <li>&gamma;</li>
 * </ul>
 * are computed from other functions, and depend on prior values. 
 * <br><br>
 * NOTE: LW is meant for algorithms that perform updates to a distance matrix. 
 * While the {@link #dissimilarity(java.util.List, java.util.List) } and 
 * {@link #dissimilarity(java.util.Set, java.util.Set, double[][]) } methods 
 * will work and produce the correct results, their performance will likely be 
 * less than desired had they be computed directly. 
 * @author Edward Raff
 */
public abstract class LanceWilliamsDissimilarity extends DistanceMetricDissimilarity implements UpdatableClusterDissimilarity
{
    /**
     * Creates a new LW dissimilarity measure using the given metric as the base distance between individual points. 
     * @param dm the base metric to measure dissimilarity from. 
     */
    public LanceWilliamsDissimilarity(DistanceMetric dm)
    {
        super(dm);
    }
    
    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public LanceWilliamsDissimilarity(LanceWilliamsDissimilarity toCopy)
    {
        this(toCopy.dm.clone());
    }
    
    /**
     * This method computes the value of the &alpha; variable. A flag is used to
     * control whether the value for the cluster <tt>i</tt> and <tt>k</tt> is 
     * being computed, or the value for the cluster <tt>j</tt> and <tt>k</tt>.
     * 
     * @param iFlag <tt>true</tt> indicates that &alpha;<sub>i</sub> is the 
     * value to compute, <tt>false</tt> indicated that &alpha;<sub>j</sub> 
     * should be computed. 
     * @param ni the number of points that make up cluster <tt>i</tt>
     * @param nj the number of points that make up cluster <tt>j</tt>
     * @param nk the number of points that make up cluster <tt>k</tt>
     * @return the value of the variable &alpha;
     */
    protected abstract double aConst(boolean iFlag, int ni, int nj, int nk);
    
    /**
     * This method computes the value of the &beta; variable.
     * @param ni the number of points that make up cluster <tt>i</tt>
     * @param nj the number of points that make up cluster <tt>j</tt>
     * @param nk the number of points that make up cluster <tt>k</tt>
     * @return the value of the variable &beta;
     */
    protected abstract double bConst(int ni, int nj, int nk);
    /**
     * This method computes the value of the &gamma; variable.
     * @param ni the number of points that make up cluster <tt>i</tt>
     * @param nj the number of points that make up cluster <tt>j</tt>
     * @param nk the number of points that make up cluster <tt>k</tt>
     * @return the value of the variable &gamma;
     */
    protected abstract double cConst(int ni, int nj, int nk);

    @Override
    public double dissimilarity(List<DataPoint> a, List<DataPoint> b)
    {
        if(a.size() == 1 && b.size() == 1)
            return dm.dist(a.get(0).getNumericalValues(), b.get(0).getNumericalValues());
        
        List<DataPoint> CI;
        List<DataPoint> CJ;
        List<DataPoint> CK;
        if(a.size() > 1)
        {
            CI = a.subList(0, 1);
            CJ = a.subList(1, a.size());
            CK = b;
        }
        else// a==1, b >1 
        {
            CI = b.subList(0, 1);
            CJ = b.subList(1, b.size());
            CK = a;
        }
            
        double d_ik = dissimilarity(CI, CK);
        double d_jk = dissimilarity(CJ, CK);
        double d_ij = dissimilarity(CI, CJ);
        return  aConst(true, CI.size(), CJ.size(), CK.size())  * d_ik + 
                aConst(false, CI.size(), CJ.size(), CK.size()) * d_jk + 
                bConst(CI.size(), CJ.size(), CK.size())        * d_ij + 
                cConst(CI.size(), CJ.size(), CK.size())        * abs(d_ik-d_jk);
    }

    @Override
    public double dissimilarity(Set<Integer> a, Set<Integer> b, double[][] distanceMatrix)
    {
        if(a.size() == 1 && b.size() == 1)
            return getDistance(distanceMatrix, getVal(a), getVal(b));
        
        Set<Integer> CI;
        Set<Integer> CJ;
        Set<Integer> CK;
        
        if(a.size() > 1)
        {
            CI = new IntSet();
            CI.add(getVal(a));
            CJ = new IntSet(a);
            CJ.removeAll(CI);
            CK = b;
        }
        else//a == 1, b > 1
        {
            CI = new IntSet();
            CI.add(getVal(b));
            CJ = new IntSet(b);
            CJ.removeAll(CI);
            CK = a;
        }
        
        double d_ik = dissimilarity(CI, CK, distanceMatrix);
        double d_jk = dissimilarity(CJ, CK, distanceMatrix);
        double d_ij = dissimilarity(CI, CJ, distanceMatrix);
        return  aConst(true, CI.size(), CJ.size(), CK.size())  * d_ik + 
                aConst(false, CI.size(), CJ.size(), CK.size()) * d_jk + 
                bConst(CI.size(), CJ.size(), CK.size())        * d_ij + 
                cConst(CI.size(), CJ.size(), CK.size())        * abs(d_ik-d_jk);
    }
    
    /**
     * Returns a value from the set, assuming that all values are positive. If empty, -1 is returned. 
     * @param a the set to get a value of
     * @return a value from the set, or -1 if empty
     */
    private static int getVal(Set<Integer> a)
    {
        for(int i : a)
            return i;
        return -1;
    }

    @Override
    public double dissimilarity(int i, int ni, int j, int nj, double[][] distanceMatrix)
    {
        return getDistance(distanceMatrix, i, j);
    }

    @Override
    public double dissimilarity(int i, int ni, int j, int nj, int k, int nk, double[][] distanceMatrix)
    {
        double d_ik = getDistance(distanceMatrix, i, k);
        double d_jk = getDistance(distanceMatrix, j, k);
        double d_ij = getDistance(distanceMatrix, i, j);
        return  dissimilarity(ni, nj, nk, d_ij, d_ik, d_jk);
    }
    
    /**
     * Provides the notion of dissimilarity between two sets of points, that may
     * not have the same number of points. This is done using a matrix 
     * containing all pairwise distance computations between all points. This 
     * distance matrix will then be updated at each iteration and merging, 
     * leaving empty space in the matrix. The updates will be done by the 
     * clustering algorithm. Implementing this interface indicates that this 
     * dissimilarity measure can be accurately computed in an updatable manner 
     * that is compatible with a Lance–Williams update. <br>
     * 
     * This computes the dissimilarity of the union of clusters i and j, 
     * (C<sub>i</sub> &cup; C<sub>j</sub>), with the cluster k. This method is 
     * used by other algorithms to perform an update of the distance matrix in 
     * an efficient manner. 
     * 
     * @param ni the number of items in the cluster represented by <tt>i</tt>
     * @param nj the number of items in the cluster represented by <tt>j</tt>
     * @param nk the number of items in the cluster represented by <tt>k</tt>
     * @param d_ij the distance between clusters i and j
     * @param d_ik the distance between clusters i and k
     * @param d_jk the distance between clusters j and k
     * @return the distance between the cluster formed from i and j, to the cluster k
     */
    public double dissimilarity(int ni, int nj, int nk, double d_ij, double d_ik, double d_jk)
    {
        return  aConst(true, ni, nj, nk)  * d_ik  +
                aConst(false, ni, nj, nk) * d_jk  + 
                bConst(ni, nj, nk)        * d_ij  + 
                cConst(ni, nj, nk)        * abs(d_ik-d_jk);
    }
    
    @Override
    abstract public LanceWilliamsDissimilarity clone();
    
}
