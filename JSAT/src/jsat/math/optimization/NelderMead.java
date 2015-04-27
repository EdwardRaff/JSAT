
package jsat.math.optimization;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.math.Function;
import jsat.utils.ProbailityMatch;

/**
 * The Nelder-Mean algorithm is a simple directed search method. As such, it does not need any information about 
 * the target functions derivative, or any data points. To perform best, the Nelder-Mean method needs N+1 
 * reasonable initial guesses for an N dimensional problem. <br>
 * The Nelder-Mean method has the advantage that the only information it needs about the function it is going to minimize, is the function itself. 
 * 
 * @author Edward Raff
 */
public class NelderMead implements Optimizer
{

	private static final long serialVersionUID = -2930235371787386607L;
	/**
     * Reflection constant
     */
    private double reflection = 1.0;
    /**
     * Expansion constant
     */
    private double expansion = 2.0;
    /**
     * Contraction constant
     */
    private double contraction = -0.5;
    /**
     * Shrink constant
     */
    private double shrink = 0.5;

    /**
     * Sets the reflection constant, which must be greater than 0
     * @param reflection the reflection constant
     */
    public void setReflection(double reflection)
    {
        if(reflection <=0 || Double.isNaN(reflection) || Double.isInfinite(reflection) )
            throw new ArithmeticException("Reflection constant must be > 0, not " + reflection);
        this.reflection = reflection;
    }

    /**
     * Sets the expansion constant, which must be greater than 1 and the reflection constant
     * @param expansion 
     */
    public void setExpansion(double expansion) 
    {
        if(expansion <= 1 || Double.isNaN(expansion) || Double.isInfinite(expansion) )
            throw new ArithmeticException("Expansion constant must be > 1, not " + expansion);
        else if(expansion <= reflection)
            throw new ArithmeticException("Expansion constant must be less than the reflection constant");
        this.expansion = expansion;
    }

    /**
     * Sets the contraction constant, which must be in the range (0, 1)
     * @param contraction the contraction constant
     */
    public void setContraction(double contraction)
    {
        if(contraction >= 1 || contraction <= 0 || Double.isNaN(contraction) || Double.isInfinite(contraction) )
            throw new ArithmeticException("Contraction constant must be > 0 and < 1, not " + contraction);
        this.contraction = contraction;
    }

    /**
     * Sets the shrinkage constant, which must be in the range (0, 1)
     * @param shrink 
     */
    public void setShrink(double shrink)
    {
        if(shrink >= 1 || shrink <= 0 || Double.isNaN(shrink) || Double.isInfinite(shrink) )
            throw new ArithmeticException("Shrinkage constant must be > 0 and < 1, not " + shrink);
        this.shrink = shrink;
    }
    
    public Vec optimize(double eps, int iterationLimit, Function f, Function fd, Vec vars, List<Vec> inputs, Vec outputs, ExecutorService threadpool)
    {
        return optimize(eps, iterationLimit, f, fd, vars, inputs, outputs);
    }

    public Vec optimize(double eps, int iterationLimit, Function f, Function fd, Vec vars, List<Vec> inputs, Vec outputs)
    {
        List<Vec> initialPoints = new ArrayList<Vec>();
        initialPoints.add(vars);
        
        return optimize(eps, iterationLimit, f, initialPoints);
    }
    
    /**
     * Attempts to find the minimal value of the given function. 
     * 
     * @param eps the desired accuracy of the result. 
     * @param iterationLimit the maximum number of iteration steps to allow. This value must be positive
     * @param f the function to optimize. This value can not be null
     * @param initalPoints the list of initial guess points. If too small, new ones will be generated. if too large, 
     * the extra ones will be ignored. This list may not be empty
     * @return the computed value for the optimization. 
     */
    public Vec optimize(double eps, int iterationLimit, Function f, List<Vec> initalPoints)
    {
        if(initalPoints.isEmpty())
            throw new ArithmeticException("Empty Initial list. Can not determin dimension of problem");
        Vec init = initalPoints.get(0);
        int N = initalPoints.get(0).length();
        //The simplex verticies paired with their value from the objective function 
        List<ProbailityMatch<Vec>> simplex = new ArrayList<ProbailityMatch<Vec>>(N);
        for(Vec vars : initalPoints)
            simplex.add(new ProbailityMatch<Vec>(f.f(vars), vars.clone()));
        Random rand = new Random(initalPoints.hashCode());
        
        while(simplex.size() < N+1)
        {
            //Better simplex geneartion?
            DenseVector newSimplex = new DenseVector(N);
            for(int i = 0; i < newSimplex.length(); i++)
                if(init.get(i) != 0)
                    newSimplex.set(i, init.get(i)*rand.nextGaussian());
                else
                    newSimplex.set(i, rand.nextGaussian());
            
            simplex.add(new ProbailityMatch<Vec>(f.f(newSimplex), newSimplex));
        }
        
        Collections.sort(simplex);
        //Remove superfolusly given points
        while(simplex.size() > N+1)
            simplex.remove(simplex.size()-1);
        
        //Center of gravity point
        Vec x0 = new DenseVector(N);
        //reflection point
        Vec xr = new DenseVector(N);
        //Extension point, also used for contraction
        Vec xec = new DenseVector(N);
        //Temp space for compuations
        Vec tmp = new DenseVector(N);
        
        final int lastIndex = simplex.size()-1;
        for(int iterationCount = 0; iterationCount < iterationLimit; iterationCount++)
        {
            //Convergence check 
            if(Math.abs(simplex.get(lastIndex).getProbability() - simplex.get(0).getProbability()) < eps)
                break;
            //Step 2: valculate x0
            x0.zeroOut();
            for(ProbailityMatch<Vec> pm : simplex)
                x0.mutableAdd(pm.getMatch());
            x0.mutableDivide(simplex.size());
            
            //Step 3: Reflection
            x0.copyTo(xr);
            x0.copyTo(tmp);
            tmp.mutableSubtract(simplex.get(lastIndex).getMatch());
            xr.mutableAdd(reflection, tmp);
            double fxr = f.f(xr);
            if(simplex.get(0).getProbability() <= fxr && fxr < simplex.get(lastIndex-1).getProbability())
            {
                insertIntoSimplex(simplex, xr, fxr);
                continue;
            }
            
            //Step 4: Expansion
            if(fxr < simplex.get(0).getProbability())//Best so far
            {
                x0.copyTo(xec);
                xec.mutableAdd(expansion, tmp);//tmp still contains (x0-xWorst)
                double fxec = f.f(xec);
                if(fxec < fxr)
                    insertIntoSimplex(simplex, xec, fxec);//Even better! Use this one
                else
                    insertIntoSimplex(simplex, xr, fxr);//Ehh, wasnt as good as we thought
                continue;
            }
            
            //Step 5: Contraction
            x0.copyTo(xec);
            xec.mutableAdd(contraction, tmp);
            double fxec = f.f(xec);
            if(fxec < simplex.get(lastIndex).getProbability())
            {
                insertIntoSimplex(simplex, xec, fxec);
                continue;
            }
            //Step 6: Reduction
            Vec xBest = simplex.get(0).getMatch();
            for(int i = 1; i < simplex.size(); i++)
            {
                ProbailityMatch<Vec> pm = simplex.get(i);
                Vec xi = pm.getMatch();
                xi.mutableSubtract(xBest);
                xi.mutableMultiply(shrink);
                xi.mutableAdd(xBest);
                pm.setProbability(f.f(xi));
            }
            Collections.sort(simplex);
        }
        
        return simplex.get(0).getMatch();
    }

    private static void insertIntoSimplex(List<ProbailityMatch<Vec>> simplex, Vec x, double fx)
    {
        //We are removing the last element and inserting a new one that is better 
        ProbailityMatch<Vec> pm = simplex.remove(simplex.size() - 1);
        pm.setProbability(fx);
        x.copyTo(pm.getMatch());

        //Now put it in the correct place
        int sortInto = Collections.binarySearch(simplex, pm);
        if (sortInto >= 0)
            simplex.add(sortInto, pm);
        else
        {
            sortInto = -(sortInto)-1;
            if(sortInto == simplex.size())//Then it was just better thne the last
                simplex.add(pm);
            else//It was a bit better then that
                simplex.add(sortInto, pm);
        }
    }
    
}
