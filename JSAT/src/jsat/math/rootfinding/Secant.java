
package jsat.math.rootfinding;

import jsat.linear.Vec;
import jsat.math.Function;

/**
 *
 * @author Edward Raff
 */
public class Secant implements RootFinder
{  

	private static final long serialVersionUID = -5175113107084930582L;

	public static double root(double a, double b, Function f, double... args)
    {
        return root(1e-15, 1000, a, b, 0, f, args);
    }
    
    public static double root(double eps, double a, double b, Function f, double... args)
    {
        return root(eps, 1000, a, b, 0, f, args);
    }
    
    public static double root(double eps, double a, double b, int pos, Function f, double... args)
    {
        return root(eps, 1000, a, b, pos, f, args);
    }
    
    public static double root(double eps, int maxIterations, double x0, double x1, int pos, Function f, double... args)
    {
        //We assume 1 dimensional function then 
        if(args == null ||args.length == 0)
        {
            pos = 0;
            args = new double[1];
        }
   
        args[pos] = x0;
        /**
         * f(x0)
         */
        double fx0 = f.f(args);
        
        while(Math.abs(x1-x0) > 2*eps && maxIterations-- > 0)
        {
            args[pos] = x1;
            
            double fx1 = f.f(args);
            
            double nextX = x1 - fx1*(x1-x0)/(fx1-fx0);
            
            x0 = x1;
            fx0 = fx1;
            x1 = nextX;
        }
        
        return x1;
    }

    public double root(double eps, int maxIterations, double[] initialGuesses, Function f, int pos, double... args)
    {
        return root(eps, maxIterations, initialGuesses[0], initialGuesses[1], pos, f, args);
    }

    public double root(double eps, int maxIterations, double[] initialGuesses, Function f, int pos, Vec args)
    {
        return root(eps, maxIterations, initialGuesses[0], initialGuesses[1], pos, f, args.arrayCopy());
    }

    public int guessesNeeded()
    {
        return 2;
    }
}
