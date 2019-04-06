/*
 * Copyright (C) 2019 Edward Raff <Raff.Edward@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package jsat.linear.vectorcollection;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.PriorityQueue;
import java.util.Set;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.math.FastMath;
import jsat.utils.ArrayUtils;
import jsat.utils.DoubleList;
import jsat.utils.IndexTable;
import jsat.utils.Pair;
import jsat.utils.Tuple3;
import jsat.utils.concurrent.ParallelUtils;

/**
 * This class implements the Dynamic Continuous Indexing algorithm for nearest
 * neighbor search in the {@link EuclideanDistance Euclidean} space only, which
 * avoids doing brute force distance computations for the majority of the
 * dataset, and requires limited memory. For k-NN search, DCI will return
 * approximately correct nearest neighbors, but the mistaken neighbors should
 * still be near the query. For radius search, DCI will return the exactly
 * correct results.<br>
 * <br>
 * See:
 * <ul>
 * <li>﻿Li, K., & Malik, J. (2017). <i>Fast k-Nearest Neighbour Search via
 * Prioritized DCI</i>. In Thirty-fourth International Conference on Machine
 * Learning (ICML). </li>
 * <li>﻿Li, K., & Malik, J. (2016). <i>Fast k-Nearest Neighbour Search via
 * Dynamic Continuous Indexing</i>. In M. F. Balcan & K. Q. Weinberger (Eds.),
 * Proceedings of The 33rd International Conference on Machine Learning (Vol.
 * 48, pp. 671–679). New York, New York, USA: PMLR.</li>
 * </ul>
 *
 * @author Edward Raff <Raff.Edward@gmail.com>
 * @param <V>
 */
public class DCI<V extends Vec> implements VectorCollection<V>
{
    private static EuclideanDistance euclid = new EuclideanDistance();
    
    /**
     * ﻿the number of simple indices ﻿that constitute a composite index
     */
    private int m;
    /**
     * ﻿the number of composite indices
     */
    private int L;
    
    /**
     * ﻿m*L random unit vectors in R^d
     */
    private Vec[][] u;
    /**
     * ﻿m*L empty binary search trees or skip lists
     */
    private NearestIterator[][] T;
    
    private List<V> vecs;
    private List<Double> cache;

    /**
     * Creates a new DCI object, that should provide relatively good result
     * quality.
     */
    public DCI()
    {
	this(15, 3);
    }

    /**
     * Creates a new DCI object, result quality depends on the number of simple
     * and composite indices
     *
     * @param m the number of simple indices per composite index (10-15 are
     *          common values)
     * @param L the number of composite indices (2-3 are common values)
     */
    public DCI(int m, int L)
    {
	this.m = m;
	this.L = L;
    }
    
    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public DCI(DCI<V> toCopy)
    {
	this.m = toCopy.m;
	this.L = toCopy.L;
	if(toCopy.u != null)
	{
	    this.u = new Vec[m][L];
	    this.T = new NearestIterator[m][L];
	    this.vecs = new ArrayList<>(toCopy.vecs);
	    this.cache = new DoubleList(toCopy.cache);
	    for(int j = 0; j < m; j++)
	    {
		for(int l = 0; l < L; l++)
		{
		    this.u[j][l] = toCopy.u[j][l].clone();
		    this.T[j][l] = toCopy.T[j][l].clone();
		}
	    }
	}
    }
    
    
    
    @Override
    public void build(boolean parallel, List<V> collection, DistanceMetric dm)
    {
	//really just checking dm == euclidean
	setDistanceMetric(dm);
	
	this.vecs = new ArrayList<>(collection);
	this.cache = euclid.getAccelerationCache(vecs, parallel);
	
	int d = collection.get(0).length();
	int n = collection.size();
	//Init u
	u = new Vec[m][L];
	for(int j = 0; j < m; j++)
	    for(int l = 0; l < L; l++)
	    {
		u[j][l] = DenseVector.random(d);
		u[j][l].mutableDivide(u[j][l].pNorm(2));
	    }
	
	//Init T
	T = new NearestIterator[m][L];
	
	
	//TODO, add more complex logic to balance parallelization over m&l loop as well as inner most loop
	//Insertions
	for(int j = 0; j < m; j++)
	{
	    for(int l = 0; l < L; l++)
	    {
		Vec u_jl = u[j][l];
		
		double[] keys = new double[n];
		int[] vals = new int[n];
		
		ParallelUtils.run(parallel, n, (start, end)->
		{
		    for(int i = start; i < end; i++)
		    {
			double p_bar = vecs.get(i).dot(u_jl);
			keys[i] = p_bar;
			vals[i] = i;
		    }
		});
		
		T[j][l] = new NearestIterator(keys, vals);
	    }
	}
	
    }

    @Override
    public void setDistanceMetric(DistanceMetric dm)
    {
	if(!(dm instanceof EuclideanDistance))
	    throw new IllegalArgumentException("DCI only works for Euclidean Distance Searches");
    }

    @Override
    public DistanceMetric getDistanceMetric()
    {
	return new EuclideanDistance();
    }

    @Override
    public void search(Vec query, double range, List<Integer> neighbors, List<Double> distances)
    {
	int n = vecs.size();
	
	int[][] C = new int[L][n];
	double[][] q_bar = new double[m][L];
	for(int j = 0; j < m; j++)
	    for(int l = 0; l < L; l++)
		q_bar[j][l] = query.dot(u[j][l]);
	List<Set<Integer>> S = new ArrayList<>();
	for(int l = 0; l < L; l++)
	    S.add(new HashSet<>());

	
	List<List<Iterator<Pair<Double, Integer>>>> q_iters = new ArrayList<>(m);
	for(int j = 0; j < m; j++)
	{
	    List<Iterator<Pair<Double, Integer>>> iter_m = new ArrayList<>(L);
	    for(int l = 0; l < L; l++)
	    {
		iter_m.add(T[j][l].nnWalk(q_bar[j][l]));
	    }
	    q_iters.add(iter_m);
	}
	
	
	///Now iterate to find indecies 
	for(int l = 0; l < L; l++)
	{
	    Set<Integer> S_l = S.get(l);
	    for(int j = 0; j < m; j++)
	    {
		Iterator<Pair<Double, Integer>> iter_jl = q_iters.get(j).get(l);
		while(iter_jl.hasNext())
		{
		    Pair<Double, Integer> pair = iter_jl.next();
		    //projection dist is a lower bound. If its > range, def not a candidate 
		    
		    double dist_lower = pair.getFirstItem()-q_bar[j][l];
		    
		    if(dist_lower > range)
			break;
		    //else, keep going
		    int indx = pair.getSecondItem();
		    C[l][indx]++;
		    if(C[l][indx] == m)//everyone agrees, you might be it
			S_l.add(indx);
		}
	    }
	}

	
	neighbors.clear();
	distances.clear();
	//the projected distance is a lower bound. So if its truley in range, 
	//it must be present in all subsets
	Map<Integer, Integer> unionCounter = new HashMap<>();
	for(Set<Integer> S_l : S)
	    for(int i : S_l)
		unionCounter.put(i, unionCounter.getOrDefault(i, 0)+1);
	
	Set<Integer> candidates = new HashSet<>();
	for(Map.Entry<Integer, Integer> entry : unionCounter.entrySet())
	    if(entry.getValue() == S.size())//you occured in every group? You are a candidate!
		candidates.add(entry.getKey());
	
	List<Double> qi = euclid.getQueryInfo(query);
	for(int i : candidates)
	{
	    neighbors.add(i);
	    distances.add(euclid.dist(i, query, qi, vecs, cache));
	}

	//sort by distance and remove excess
	IndexTable it = new IndexTable(distances);
	it.apply(neighbors);
	it.apply(distances);
	
	int maxIndx = ArrayUtils.bsIndex2Insert(Collections.binarySearch(distances, range));
	

	neighbors.subList(maxIndx, neighbors.size()).clear();
	distances.subList(maxIndx, distances.size()).clear();
    }

    @Override
    public void search(Vec query, int numNeighbors, List<Integer> neighbors, List<Double> distances)
    {
	int n = vecs.size();
	int k1 = (int) (m*numNeighbors*(FastMath.log(n)-FastMath.log(numNeighbors)));
	
	int[][] C = new int[L][n];
	double[][] q_bar = new double[m][L];
	for(int j = 0; j < m; j++)
	    for(int l = 0; l < L; l++)
		q_bar[j][l] = query.dot(u[j][l]);
	List<Set<Integer>> S = new ArrayList<>();
	for(int l = 0; l < L; l++)
	    S.add(new HashSet<>());

	
	List<List<Iterator<Pair<Double, Integer>>>> q_iters = new ArrayList<>(m);
	for(int j = 0; j < m; j++)
	{
	    List<Iterator<Pair<Double, Integer>>> iter_m = new ArrayList<>(L);
	    for(int l = 0; l < L; l++)
	    {
		iter_m.add(T[j][l].nnWalk(q_bar[j][l]));
	    }
	    q_iters.add(iter_m);
	}
	
	//Prep priority Qs
	/**
	 * First value is the priority
	 * second value is the index j in [0, m) that it came from
	 * third value is the index i in the vector array of the point being referenced
	*/
	List<PriorityQueue<Tuple3<Double, Integer, Integer>>> P = new ArrayList<>();
	for(int l = 0; l < L; l++)
	    P.add(new PriorityQueue<>((o1, o2) -> Double.compare(o1.getX(), o2.getX())));
	for(int j = 0; j < m; j++)
	    for(int l = 0; l < L; l++)
	    {
		Pair<Double, Integer> ph = q_iters.get(j).get(l).next();
		double priority = Math.abs(ph.getFirstItem()-q_bar[j][l]);
		P.get(l).add(new Tuple3<>(priority, j, ph.getSecondItem()));
	    }
	
	///Now iterate to find indecies 
	
	for(int i = 0; i < k1; i++)
	{
	    for(int l = 0; l < L; l++)
	    {
		Set<Integer> S_l = S.get(l);
		PriorityQueue<Tuple3<Double, Integer, Integer>> P_l = P.get(l);
		if(S_l.size() < numNeighbors)
		{
		    Tuple3<Double, Integer, Integer> ph = P_l.poll();
		    int j = ph.getY();
		    int h_jl =  ph.getZ();
		    
		    Pair<Double, Integer> next_ph = q_iters.get(j).get(l).next();
		    double priority = Math.abs(next_ph.getFirstItem()-q_bar[j][l]);
		    P.get(l).add(new Tuple3<>(priority, j, next_ph.getSecondItem()));
		    
		    C[l][h_jl]++;
		    
		    if(C[l][h_jl] == m)
			S_l.add(h_jl);
		}
	    }
	    
	    //We haven't even found as many candidates as we have neighbors we are looking for? Up the iterations then!
	    if(i == k1-1 && S.stream().mapToInt(s->s.size()).min().getAsInt() < numNeighbors)
		k1 *= 2;
	}
	
	neighbors.clear();
	distances.clear();
	Set<Integer> candidates = new HashSet<>();
	for(Set<Integer> S_l : S)
	    candidates.addAll(S_l);
	
	List<Double> qi = euclid.getQueryInfo(query);
	for(int i : candidates)
	{
	    neighbors.add(i);
	    distances.add(euclid.dist(i, query, qi, vecs, cache));
	}

	//sort by distance and remove excess
	IndexTable it = new IndexTable(distances);
	it.apply(neighbors);
	it.apply(distances);

	neighbors.subList(numNeighbors, neighbors.size()).clear();
	distances.subList(numNeighbors, distances.size()).clear();
    }

    @Override
    public V get(int indx)
    {
	return vecs.get(indx);
    }

    @Override
    public List<Double> getAccelerationCache()
    {
	return cache;
    }

    @Override
    public int size()
    {
	return vecs.size();
    }

    @Override
    public DCI<V> clone()
    {
	return new DCI<>(this);
    }

    /**
     * We need to be able to store a pair of tuples <Double, Integer>, and given
     * a query double q, iterate through the points in the collection based on
     * which tuples are closest to the query. TreeMap dosn't let us do this. So
     * custom class to implement the logic in a compact manner as arrays.
     */
    static class NearestIterator
    {
	public double[] keys;
	public int[] vals;

	public NearestIterator(double[] keys, int[] vals)
	{
	    this.keys = keys;
	    this.vals = vals;
	    
	    if(keys.length != vals.length)
		throw new IllegalArgumentException("Keys and vales should have the same length");
	    
	    IndexTable it = new IndexTable(keys);
	    it.apply(keys);
	    it.apply(vals);
	}

	public NearestIterator(NearestIterator toCopy)
	{
	    this.keys = Arrays.copyOf(toCopy.keys, toCopy.keys.length);
	    this.vals = Arrays.copyOf(toCopy.vals, toCopy.vals.length);
	}

	public NearestIterator()
	{
	}


	@Override
	protected NearestIterator clone() 
	{
	    return new NearestIterator(this);
	}
	
	public Iterator<Pair<Double, Integer>> nnWalk(double q)
	{

	    return new Iterator<Pair<Double, Integer>>()
	    {
		int upper = ArrayUtils.bsIndex2Insert(Arrays.binarySearch(keys, q));
		//upper is now the lowest index of a point that is >= q
		int lower = upper-1;
		
		@Override
		public boolean hasNext()
		{
		    return lower >= 0 || upper < keys.length;
		}

		@Override
		public Pair<Double, Integer> next()
		{
		    Pair<Double, Integer> toRet = null;
		    if (lower < 0 && upper >= keys.length)
		    {
			throw new NoSuchElementException();
		    }
		    else if (lower < 0)//upper is only option
		    {
			toRet = new Pair<>(keys[upper], vals[upper]);
			upper++;
		    }
		    else if (upper >= keys.length)//lower is only options
		    {
			toRet = new Pair<>(keys[lower], vals[lower]);
			lower--;
		    }
		    else if (Math.abs(keys[upper] - q) < Math.abs(keys[lower] - q))
		    {//upper is closer to q, so return that
			toRet = new Pair<>(keys[upper], vals[upper]);
			upper++;
		    }
		    else//lower must be closer
		    {
			toRet = new Pair<>(keys[lower], vals[lower]);
			lower--;
		    }
		    return toRet;
		}
	    };
	}
	
	
    }
    
    
}
