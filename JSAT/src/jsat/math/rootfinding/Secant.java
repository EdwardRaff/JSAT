
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

	public static double root(final double a, final double b, final Function f, final double... args)
    {
        return root(1e-15, 1000, a, b, 0, f, args);
    }
    
    public static double root(final double eps, final double a, final double b, final Function f, final double... args)
    {
        return root(eps, 1000, a, b, 0, f, args);
    }
    
    public static double root(final double eps, final double a, final double b, final int pos, final Function f, final double... args)
    {
        return root(eps, 1000, a, b, pos, f, args);
    }
    
    public static double root(final double eps, int maxIterations, double x0, double x1, int pos, final Function f, double... args)
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
            
            final double fx1 = f.f(args);
            
            final double nextX = x1 - fx1*(x1-x0)/(fx1-fx0);
            
            x0 = x1;
            fx0 = fx1;
            x1 = nextX;
        }
        
        return x1;
    }

    public double root(final double eps, final int maxIterations, final double[] initialGuesses, final Function f, final int pos, final double... args)
    {
        return root(eps, maxIterations, initialGuesses[0], initialGuesses[1], pos, f, args);
    }

    public double root(final double eps, final int maxIterations, final double[] initialGuesses, final Function f, final int pos, final Vec args)
    {
        return root(eps, maxIterations, initialGuesses[0], initialGuesses[1], pos, f, args.arrayCopy());
    }

    public int guessesNeeded()
    {
        return 2;
    }
}
