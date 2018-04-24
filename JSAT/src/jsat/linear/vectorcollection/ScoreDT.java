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

/**
 *
 * @author Edward Raff
 */
public interface ScoreDT
{
    /**
     * 
     * @param query
     * @param ref
     * @return {@link Double#POSITIVE_INFINITY} if the node should be pruned. 
     */
    public double score(IndexNode ref, IndexNode query);
    
    /**
     * This method re-scores a given reference query node pair. By default this
     * simply returns the original score that was given and does no computation.
     * If the given original score does not look valid (is less than zero), the
     * score will be re-computed. Some algorithms may choose to implement this
     * method when pruning is best done after initial depth-first traversals
     * have already been completed of other branches.
     *
     * @param ref
     * @param query
     * @param origScore
     * @return 
     */
    default double score(IndexNode ref, IndexNode query, double origScore)
    {
        if(origScore < 0)
            return score(ref, query);
        else
            return origScore;
    }
    
}
