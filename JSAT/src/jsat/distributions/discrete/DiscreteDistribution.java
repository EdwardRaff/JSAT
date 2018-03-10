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
import jsat.math.Function1D;
import jsat.math.rootfinding.Zeroin;

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
    public double invCdf(double p)
    {
        return invCdfRootFinding(p, 1e-6);
    }
    
    /**
     * Helper method that computes the inverse CDF by performing root-finding on
     * the CDF of the function. This provides a convenient default method for
     * any invCdfRootFinding implementation, but may not be as fast or accurate
     * as possible.
     *
     * @param p the probability value
     * @param tol the search tolerance
     * @return the value such that the CDF would return p
     */
    protected double invCdfRootFinding(double p, double tol)
    {
        if (p < 0 || p > 1)
            throw new ArithmeticException("Value of p must be in the range [0,1], not " + p);
        //two special case checks, as they can cause a failure to get a positive and negative value on the ends, which means we can't do a search for the root
        //Special case check, p < min value
        if(min() >= Integer.MIN_VALUE)
            if(p <= cdf(min()))
                return min();
        //special case check, p >= max value
        if(max() < Integer.MAX_VALUE)
            if(p > cdf(max()-1))
                return max();
        //stewpwise nature fo discrete can cause problems for search, so we will use a smoothed cdf to pass in
//        double toRet= invCdf(p, );

        //Lets use an interpolated version of the CDF so that our numerical methods will behave better
        Function1D cdfInterpolated = (double x) ->
        {
            double query = x;
            //if it happens to fall on an int we just compute the regular value
            if(Math.rint(query) == query)
                return cdf((int)query) - p;
            //else, interpolate
            double larger = query+1;
            double diff = larger-query;
            return cdf(query)*diff + cdf(larger)*(1-diff) - p;
        };
        
        
        double a = Double.isInfinite(min()) ? Integer.MIN_VALUE*.95 : min();
        double b = Double.isInfinite(max()) ? Integer.MAX_VALUE*.95 : max();
        
        double toRet = Zeroin.root(tol, a, b, cdfInterpolated);
        return Math.round(toRet);
    }

    
    @Override
    abstract public DiscreteDistribution clone();
    
    
    
}

