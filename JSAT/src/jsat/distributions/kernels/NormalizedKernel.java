/*
 * Copyright (C) 2016 Edward Raff
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
package jsat.distributions.kernels;

import java.util.Arrays;
import java.util.List;
import jsat.linear.Vec;
import jsat.parameters.Parameter;

/**
 * This provides a wrapper kernel that produces a normalized kernel trick from
 * any input kernel trick. A normalized kernel has a maximum output of 1 when
 * two inputs are the same.
 *
 * @author Edward Raff
 */
public class NormalizedKernel implements KernelTrick
{
    private KernelTrick k;

    public NormalizedKernel(KernelTrick source_kernel)
    {
        this.k = source_kernel;
    }

    @Override
    public NormalizedKernel clone()
    {
        return new NormalizedKernel(k.clone());
    }

    @Override
    public double eval(Vec a, Vec b)
    {
        double aa = k.eval(a, a);
        double bb = k.eval(b, b);
        if(aa == 0 || bb == 0)
            return 0;
        else
            return k.eval(a, b)/Math.sqrt(aa*bb);
    }

    @Override
    public List<Parameter> getParameters()
    {
        return k.getParameters();
    }

    @Override
    public Parameter getParameter(String paramName)
    {
        return k.getParameter(paramName);
    }

    @Override
    public boolean supportsAcceleration()
    {
        return k.supportsAcceleration();
    }

    @Override
    public List<Double> getAccelerationCache(List<? extends Vec> trainingSet)
    {
        return k.getAccelerationCache(trainingSet);
    }

    @Override
    public List<Double> getQueryInfo(Vec q)
    {
        return k.getQueryInfo(q);
    }

    @Override
    public void addToCache(Vec newVec, List<Double> cache)
    {
        k.addToCache(newVec, cache);
    }

    @Override
    public double eval(int a, Vec b, List<Double> qi, List<? extends Vec> vecs, List<Double> cache)
    {
        double aa = k.eval(a, a, vecs, cache);
        double bb = k.eval(0, 0, Arrays.asList(b), qi);
        if(aa == 0 || bb == 0)
            return 0;
        else
            return k.eval(a, b, qi, vecs, cache)/Math.sqrt(aa*bb);
    }

    @Override
    public double eval(int a, int b, List<? extends Vec> trainingSet, List<Double> cache)
    {
        double aa = k.eval(a, a, trainingSet, cache);
        double bb = k.eval(b, b, trainingSet, cache);
        if(aa == 0 || bb == 0)
            return 0;
        else
            return k.eval(a, b, trainingSet, cache)/Math.sqrt(aa*bb);
    }

    @Override
    public double evalSum(List<? extends Vec> finalSet, List<Double> cache, double[] alpha, Vec y, int start, int end)
    {
        return evalSum(finalSet, cache, alpha, y, getQueryInfo(y), start, end);
    }

    @Override
    public double evalSum(List<? extends Vec> finalSet, List<Double> cache, double[] alpha, Vec y, List<Double> qi, int start, int end)
    {
        double sum = 0;
        
        for(int i = start; i < end; i++)
            sum += alpha[i] * eval(i, y, qi, finalSet, cache);
        
        return sum;
    }
    
    @Override
    public boolean normalized()
    {
        return true;
    }
}
