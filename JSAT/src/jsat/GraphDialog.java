

package jsat;

import java.awt.Frame;
import java.awt.GridLayout;
import javax.swing.JFrame;
import jsat.graphing.Graph2D;

/**
 *
 * @author Edward Raff
 */
public class GraphDialog extends JFrame
{
    Graph2D graph;

    public GraphDialog(Frame parent, String title, Graph2D graph)
    {
        super(title);
        this.graph = graph;


        getContentPane().setLayout(new GridLayout(1, 1));
        getContentPane().add(graph);
    }


}
