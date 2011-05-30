package jsat.utils;

/**
 * Class allows the arbitrary association of some object type with a probability. 
 * @author Edward Raff
 */
public class ProbailityMatch<T> implements Comparable<ProbailityMatch>
{

    private double probability;
    private T match;

    public ProbailityMatch(double probability, T match)
    {
        this.probability = probability;
        this.match = match;
    }

    public int compareTo(ProbailityMatch t)
    {
        return new Double(probability).compareTo(t.probability);
    }

    public double getProbability()
    {
        return probability;
    }

    public void setProbability(double probability)
    {
        this.probability = probability;
    }

    public T getMatch()
    {
        return match;
    }

    public void setMatch(T match)
    {
        this.match = match;
    }
}
