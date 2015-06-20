/*
 * Copyright (C) 2015 Edward Raff
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
package jsat.distributions.discrete;

import jsat.distributions.Distribution;

/**
 * This abstract class defines the contract for a distribution over the integer
 * values.<br>
 * <br>
 * The {@link #cdf(double) } method will behave by
 * {@link Math#floor(double) rounding down} and then calling the integer {@link #cdf(int)
 * } counterpart.
 *
 * @author Edward Raff
 */
abstract public class DiscreteDistribution extends Distribution
{
    /**
     * Computes the log of the Probability Mass Function. Note, that then the
     * probability is zero, {@link Double#NEGATIVE_INFINITY} would be the true
     * value. Instead, this method will always return the negative of
     * {@link Double#MAX_VALUE}. This is to avoid propagating bad values through
     * computation.
     *
     * @param x the value to get the log(PMF) of
     * @return the value of log(PMF(x))
     */
    public double logPmf(int x)
    {
        double pmf = pmf(x);
        if (pmf <= 0)
            return -Double.MAX_VALUE;
        return Math.log(pmf);
    }

    abstract public double pmf(int x);
    
    /**
     * Computes the value of the Cumulative Density Function (CDF) at the given point. 
     * The CDF returns a value in the range [0, 1], indicating what portion of values 
     * occur at or below that point. 
     * 
     * @param x the value to get the CDF of
     * @return the CDF(x) 
     */
    abstract public double cdf(int x);

    @Override
    public double cdf(double x)
    {
        return cdf((int)Math.floor(x));
    }
    
    @Override
    abstract public DiscreteDistribution clone();
    
    
    
}

