import os

from PIL import Image, ImageDraw

class TreeDrawing:

    @staticmethod
    def _getwidth(node):
        if not node.nodes_from:
            return 1
        else:
            result = 0
            for i in range(0, len(node.nodes_from)):
                result += TreeDrawing._getwidth(node.nodes_from[i])
            return result

    @staticmethod
    def _draw_node(node, draw, x, y):
        if node.nodes_from:
            allwidth = 0
            for c in node.nodes_from:
                allwidth += TreeDrawing._getwidth(c) * 100
            left = x - allwidth / 2
            # draw the function name
            draw.text((x - 10, y - 10), node.eval_strategy.model.__class__.__name__, (0, 0, 0))

            # draw the children
            for c in node.nodes_from:
                wide = TreeDrawing._getwidth(c) * 100
                draw.line((x, y, left + wide / 2, y + 100), fill=(255, 0, 0))
                TreeDrawing._draw_node(c, draw, left + wide / 2, y + 100)
                left = left + wide
        else:
            draw.text((x - 5, y), str(node.eval_strategy.model.__class__.__name__), (0, 0, 0))

    @staticmethod
    def draw_branch(node, path, generation_num= None, ind_number = None, ind_fitness=None, tree_layer= None, model_name=None, before_mutation=True):


        file_name = TreeDrawing.name_generator(path,generation_num,ind_number,ind_fitness, tree_layer, model_name, before_mutation)

        w = TreeDrawing._getwidth(node) * 100
        if not node.nodes_from:
            h = 100 + 120
        else:
            h = node.get_depth_down() * 100 + 120

        img = Image.new('RGB', (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        TreeDrawing._draw_node(node, draw, w / 2, 20)
        img.save(file_name, 'PNG')


    @staticmethod
    def name_generator(path, generation_num= None, ind_number = None, ind_fitness = None,tree_layer= None, model_name=None, before_mutation=True):

        if not os.path.isdir(f'../../tmp'):
            os.mkdir(f'../../tmp')
        if not os.path.isdir(f'../../tmp/Trees'):
            os.mkdir(f'../../tmp/Trees')

        if path =="population":
            if not os.path.isdir(f'../../tmp/Trees/population'):
                os.mkdir(f'../../tmp/Trees/population')
            if not os.path.isdir(f'../../tmp/Trees/population/pop{generation_num}'):
                os.mkdir(f'../../tmp/Trees/population/pop{generation_num}')
            return f'../../tmp/Trees/population/pop{generation_num}/{ind_number}(fitness_{ind_fitness}).png'

        if path =="best_individuals":
            if not os.path.isdir(f'../../tmp/Trees/best_individuals'):
                os.mkdir(f'../../tmp/Trees/best_individuals')

            return f'../../tmp/Trees/best_individuals/the_best_ind_(pop{generation_num}).png'

        if path =="crossover":
            if not os.path.isdir(f'../../tmp/Trees/crossover'):
                os.mkdir(f'../../tmp/Trees/crossover')
            if tree_layer:
                return f'../../tmp/Trees/crossover/p1_pair{ind_number}_pop{generation_num}_rnlayer{tree_layer}({model_name}).png'
            else:
                return f'../../tmp/Trees/crossover/result_pair{ind_number}_pop{generation_num}.png'

        if path =="mutation":
            if not os.path.isdir(f'../../tmp/Trees/mutation'):
                os.mkdir(f'../../tmp/Trees/mutation')
            condition_in_mut = "before_mut" if before_mutation else "after_mut"
            return f'../../tmp/Trees/mutation/{condition_in_mut}_pop{generation_num}_ind{ind_number}.png'


