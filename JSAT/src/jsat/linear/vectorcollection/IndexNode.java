/*
 * Copyright (C) 2018 Edward Raff
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
import java.util.List;
import java.util.PriorityQueue;
import jsat.linear.Vec;
import static java.lang.Math.*;
/**
 *
 * @author Edward Raff
 * @param <N>
 */
public interface IndexNode<N extends IndexNode>
{
    
    /**
     * 
     * @return returns the parent node to this one, or {@code null} if this node is the root. 
     */
    public N getParrent();


    /**
     * This method returns a lower bound on the minimum distance from a point in
     * or owned by this node to any point in or owner by the {@code other} node.
     * Because the value return is a lower bound, 0 would always be a valid
     * return value.
     *
     * @param other the other node of points to get the minimum distance to
     * @return a lower bound on the minimum distance.
     */
    public double minNodeDistance(N other);
    
    public double maxNodeDistance(N other);
    
    /**
     * This method returns a lower bound on the minimum distance from a point in
     * or owned by this node to the  {@code other} point.
     * Because the value return is a lower bound, 0 would always be a valid
     * return value.
     * @param other
     * @return 
     */
    public double minNodeDistance(int other);
    
    /**
     * Gets the distance from this node (or its centroid) to it's parent node (or centroid). 
     * @return 
     */
    public default double getParentDistance()
    {
        N parent = getParrent();
        if(parent == null)
            return 0;//You have no parent
        else
            return parent.furthestDescendantDistance();//stupid loose default bound
    }
    
    /**
     * Returns an upper bound on the farthest distance from this node to any of the points it owns. <br>
     * <br>
     * In the Dual Tree papers, this is often given as ρ(N_i)
     * @return an upper bound on the distance
     */
    public double furthestPointDistance();
    
    /**
     * Returns an upper bound on the farthest distance from this node to any of
     * the points it owns or its children own. <br>
     * <br> In the Dual Tree papers,
     * this is often given as λ(Ni)
     * @return an upper bound on the distance
     */
    public double furthestDescendantDistance();
    
    
        
    /**
     * 
     * @return 
     */
    public int numChildren();
    
    public IndexNode getChild(int indx);
    
    public Vec getVec(int indx);
    
    public int numPoints();
    
    public int getPoint(int indx);
    
    public default boolean hasChildren()
    {
        return numChildren() > 0;
    }

    default public boolean allPointsInLeaves()
    {
        return true;
    }
    
    public static void dual_depth_first(IndexNode n_r, IndexNode n_q, BaseCaseDT base, ScoreDT score, boolean improvedSearch)
    {
        //Algo 10 in Thesis
        
        //3: {Perform base cases for points in node combination.}
        for(int i = 0; i < n_r.numPoints(); i++)
            for(int j = 0; j < n_q.numPoints(); j++)
                base.base_case(n_r.getPoint(i), n_q.getPoint(j));
        
        //7: {Assemble list of combinations to recurse into.}
        //8: q←empty priority queue
        PriorityQueue<IndexTuple> q = new PriorityQueue<>();
        
        //9: if Nq andNr both have children then
        if(n_q.hasChildren() && n_r.hasChildren())
        {
            //the Algorithm 10 version. Simpler but not as efficent
            if(!improvedSearch)
            {
                for(int i = 0; i < n_r.numChildren(); i++)
                    for(int j = 0; j < n_q.numChildren(); j++)
                    {
                        IndexNode n_r_i = n_r.getChild(i);
                        IndexNode n_q_j = n_q.getChild(j);

                        double s = score.score(n_r_i, n_q_j);
                        if(!Double.isNaN(s))
                            q.offer(new IndexTuple(n_r_i, n_q_j, s));
                    }
            }
            else //Below is the Algo 13 version. 
            {
                for(int c = 0; c < n_q.numChildren(); c++)
                {
                    IndexNode n_q_c = n_q.getChild(c);
                    List<IndexTuple> q_qc =new ArrayList<>();
                    boolean all_scores_same = true;
                    for(int i = 0; i < n_r.numChildren(); i++)
                    {
                        IndexNode n_r_i = n_r.getChild(i);
                        double s = score.score(n_r_i, n_q_c);
                        //check if all scores have the same value
                        if(i > 0 && abs(q_qc.get(i-1).priority-s) < 1e-13)
                            all_scores_same = false;
                        q_qc.add(new IndexTuple(n_r_i, n_q_c, s));
                    }

                    if(all_scores_same)
                    {
                        double s = score.score(n_r, n_q_c);
                        q.offer(new IndexTuple(n_r, n_q_c, s));
                    }
                    else
                        q.addAll(q_qc);
                }
            }
        }
        else if(n_q.hasChildren()) //implicitly n_r has not children if this check passes
        {
            for(int j = 0; j < n_q.numChildren(); j++)
            {
                IndexNode n_q_j = n_q.getChild(j);
                double s = score.score(n_r, n_q_j);
                if (!Double.isNaN(s))
                    q.offer(new IndexTuple(n_r, n_q_j, s));
            }
        }
        else if(n_r.hasChildren())// implicitly n_q has no children if this check passes
        {
            for (int i = 0; i < n_r.numChildren(); i++)
            {
                IndexNode n_r_i = n_r.getChild(i);
                double s = score.score(n_r_i, n_q);
                if (!Double.isNaN(s))
                    q.offer(new IndexTuple(n_r_i, n_q, s));
            }
        }
        
        
        //22: {Recurse into combinations with highest priority first.
        while(!q.isEmpty())
        {
            IndexTuple toProccess = q.poll();
            dual_depth_first(toProccess.a, toProccess.b, base, score, improvedSearch);
        }
    }
            
}

