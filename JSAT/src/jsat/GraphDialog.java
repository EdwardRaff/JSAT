

package jsat;

import java.awt.Frame;
import java.awt.GridLayout;
import javax.swing.JDialog;
import jsat.graphing.Graph2D;

/**
 *
 * @author Edward Raff
 */
public class GraphDialog extends JDialog
{
    Graph2D graph;

    public GraphDialog(Frame parent, String title, Graph2D graph)
    {
        super(parent, title, false);
        this.graph = graph;

        getContentPane().setLayout(new GridLayout(1, 1));
        getContentPane().add(graph);
    }


}
