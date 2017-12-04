package jsat.math.optimization;

import jsat.math.Function1D;

/**
 * Minimizes a single variate function in the same way that
 * 
 * @author Edward Raff
 */
public class GoldenSearch
{
    /**
     * Phi (golden ratio) minus 1
     */
    private static final double tau = (Math.sqrt(5.0) - 1.0)/2.0;
    private static final double om_tau = 1-tau;
    
    /**
     * Finds the local minimum of the function {@code f}. 
     * @param eps the desired accuracy of the result
     * @param maxIterations the maximum number of iterations to perform
     * @param a the left bound on the minimum
     * @param b the right bound on the minimum
     * @param f the function to find the minimize of

     * @return the value of variable {@code pos} that produces the local minima
     */
    public static double minimize(double eps, int maxIterations, double a, double b, Function1D f)
    {
        if (a > b)
        {
            double tmp = b;
            b = a;
            a = tmp;
        }

        //Intitial values
        int iter = 0;
        
        double x1 = a + om_tau*(b-a);
        double f1 = f.f(x1);
        
        double x2 = a + tau*(b-a);
        double f2 = f.f(x2);
        
        while (b - a > 2 * eps && iter < maxIterations)
        {
            if(f1 > f2)
            {
                a = x1;
                x1 = x2;
                f1 = f2;
                x2 = a + tau*(b-a);
                f2 = f.f(x2);
            }
            else//f1 < f2
            {
                b = x2;
                x2 = x1;
                f2 = f1;
                x1 = a + om_tau*(b-a);
                f1 = f.f(x1);
            }
            iter++;
        }
        
        return (a + b) / 2.0;
    }
}
