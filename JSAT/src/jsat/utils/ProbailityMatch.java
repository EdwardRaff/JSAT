package jsat.utils;

import java.io.Serializable;

/**
 * Class allows the arbitrary association of some object type with a probability. 
 * @author Edward Raff
 */
public class ProbailityMatch<T> implements Comparable<ProbailityMatch<T>>, Serializable
{


	private static final long serialVersionUID = -1544116376166946986L;
	private double probability;
    private T match;

    public ProbailityMatch(final double probability, final T match)
    {
        this.probability = probability;
        this.match = match;
    }

  @Override
    public int compareTo(final ProbailityMatch t)
    {
        return new Double(probability).compareTo(t.probability);
    }

    public double getProbability()
    {
        return probability;
    }

    public void setProbability(final double probability)
    {
        this.probability = probability;
    }

    public T getMatch()
    {
        return match;
    }

    public void setMatch(final T match)
    {
        this.match = match;
    }
}
