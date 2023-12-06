#include <stdio.h> 
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "list.h"

#ifdef UNIT_TESTING
/* Redirect printf to a function in the test application so it's possible to
 * test the standard output. You can ignore this; it's not relevant to the
 * assignment. */
#ifdef printf
#undef printf
#endif /* printf */
extern int test_printf(const char *format, ...);
#define printf test_printf
#endif

// Node of the singly linked list
typedef struct _node {
    char* item_name;
    float price;
    int quantity;
    struct _node *next;
} node;


#define MAX_ITEM_PRINT_LEN 100

// Note: All list_ functions should return a status code
// EXIT_FAILURE or EXIT_SUCCESS to indicate whether the operation was 
// successful or not.

// create a new list
int list_init(node **head)
{
    if (head == NULL)
    {
        return EXIT_FAILURE;
    }
    *head = NULL;
    return EXIT_SUCCESS;
}

// print a single list item to an externally allocated string
// This should be in the format of:
// "quantity * item_name @ $price ea", where item_name is a string and 
// price is a float formatted with 2 decimal places.
int list_item_to_string(node *head, char *str) {
    if (head == NULL || str == NULL) {
        return EXIT_FAILURE;
    }
    int n = snprintf(str, MAX_ITEM_PRINT_LEN, "%d * %s @ $%.2f ea", head->quantity, head->item_name, head->price);
    if (n < 0 || n >= MAX_ITEM_PRINT_LEN) {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

// print the list to stdout
// This should be in the format of:
// "pos: quantity * item_name @ $price ea", where 
//   pos is the position of the item in the list, 
//   item_name is the item_name of the item and 
//   price is the float price of the item formatted with 2 decimal places.
// For example:
// """1: 3 * banana @ $1.00 ea
// 2: 2 * orange @ $2.00 ea
// 3: 4 * apple @ $3.00 ea
// """
// It should return a newline character at the end of each item. 
// It should not have a leading newline character.
int list_print(node *head) {
    if (head == NULL) {
        return EXIT_FAILURE;
    }
    node *curr = head;
    int pos = 1;
    while (curr != NULL) {
        char str[MAX_ITEM_PRINT_LEN];
        printf("%d: ", pos);
        if (list_item_to_string(curr, str) == EXIT_FAILURE) {
            return EXIT_FAILURE;
        }
        printf("%s\n", str);
        curr = curr->next;
        pos++;
    }
    return EXIT_SUCCESS;
}

// add a new item (name, price, quantity) to the list at position pos, 
//   such that the added item is the item at position pos
// For example:
// If the list is:
// 1: 3 * banana @ $1.00 ea
// 2: 2 * orange @ $2.00 ea
// and you call list_add_item_at_pos(&head, "apple", 3.0, 4, 2)
// the list should be:
// 1: 3 * banana @ $1.00 ea
// 2: 4 * apple @ $3.00 ea
// 3: 2 * orange @ $2.00 ea
int list_add_item_at_pos(node **head, char *item_name, float price, int quantity, unsigned int pos)
{
    if (head == NULL || item_name == NULL) {
        return EXIT_FAILURE;
    }

    int i = 1;
    node *curr = *head;
    node *prev = NULL;
    while (curr != NULL && i < pos) {
        prev = curr;
        curr = curr->next;
        i++;
    }

    // if we couldn't find the right position, return failure
    if (i != pos) {
        return EXIT_FAILURE;
    }
    
    node *new_node = malloc(sizeof(node));
    if (new_node == NULL) {
        return EXIT_FAILURE;
    }
    new_node->item_name = strdup(item_name);
    new_node->price = price;
    new_node->quantity = quantity;
    new_node->next = curr;
    if (prev == NULL) {
        *head = new_node;
    } else {
        prev->next = new_node;
    }
    return EXIT_SUCCESS;
}

// helper. if previous is non-null, it will be set to point to the item at pos-1
static node* find_node_at_pos(node *head, unsigned int pos, node **previous) {
    if (head == NULL) {
        return NULL;
    }
    int i = 1;
    node *curr = NULL;
    node *prev = NULL;
    for (curr = head; curr != NULL && i < pos; curr = curr->next, i++) {
        prev = curr;
        // do nothing
    }
    if (i == pos) {
        if (previous) *previous = prev;
        return curr;
    }
    else {
        if (previous) *previous = NULL;
        return NULL;
    }
}

// update the item at position pos
int list_update_item_at_pos(node **head, char *item_name, float price, int quantity, unsigned int pos) {
    if (head == NULL || item_name == NULL) {
        return EXIT_FAILURE;
    }
    node *curr = find_node_at_pos(*head, pos, NULL);
    if (curr == NULL) {
        return EXIT_FAILURE;
    }
    if (curr->item_name != NULL) {
        free(curr->item_name);
    }
    curr->item_name = strdup(item_name);
    curr->price = price;
    curr->quantity = quantity;
    return EXIT_SUCCESS;
}

// remove the item at position pos
int list_remove_item_at_pos(node **head, int pos)
{    
    if (head == NULL) {
        return EXIT_FAILURE;
    }
    node *prev = NULL;
    node *to_delete = find_node_at_pos(*head, pos, &prev);
    if (to_delete == NULL) {
        return EXIT_FAILURE;
    }
    if (prev == NULL) {
        *head = to_delete->next;
    } else {
        prev->next = to_delete->next;
    }
    free(to_delete->item_name);
    free(to_delete);
    return EXIT_SUCCESS;
}

// swap the item at position pos1 with the item at position pos2
int list_swap_item_positions(node **head, int pos1, int pos2) {
    if (head == NULL) {
        return EXIT_FAILURE;
    }
    
    node *prev = NULL;
    node *curr = find_node_at_pos(*head, pos1, &prev);
    if (curr == NULL) {
        return EXIT_FAILURE;
    }
    node *prev2 = NULL;
    node *curr2 = find_node_at_pos(*head, pos2, &prev2);
    if (curr2 == NULL) {
        return EXIT_FAILURE;
    }
    // Early exit if items to swap are the same
    if (curr == curr2) return EXIT_SUCCESS;
    
    // Swap
    if (prev == NULL) {
        *head = curr2;
    } else {
        prev->next = curr2;
    }
    if (prev2 == NULL) {
        *head = curr;
    } else {
        prev2->next = curr;
    }
    node *temp = curr->next;
    curr->next = curr2->next;
    curr2->next = temp;

    return EXIT_SUCCESS;
}

// find the item position with the highest single price
int list_find_highest_price_item_position(node *head, int *pos) 
{
    if (head == NULL || pos == NULL) {
        return EXIT_FAILURE;
    }
    int i = 1;
    node *curr = head;
    float max = -INFINITY;
    while (curr != NULL) {
        if (curr->price > max) {
            max = curr->price;
            *pos = i;
        }
        curr = curr->next;
        i++;
    }
    return EXIT_SUCCESS;
}

// calculate the total cost of the list (sum of all prices * quantities)
int list_cost_sum(node *head, float *total)
{
    if (head == NULL || total == NULL) {
        return EXIT_FAILURE;
    }
    float sum = 0;
    node *curr = head;
    while (curr != NULL) {
        sum += curr->price * curr->quantity;
        curr = curr->next;
    }
    *total = sum;
    return EXIT_SUCCESS;
}

// save the list to file filename
// the file should be in the following format:
// item_name,price,quantity\n 
//   (one item per line, separated by commas, and newline at the end)
int list_save(node *head, char *filename)
{
    if (head == NULL || filename == NULL) {
        return EXIT_FAILURE;
    }
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        return EXIT_FAILURE;
    }
    node *curr = head;
    while (curr != NULL) {
        int n = fprintf(fp, "%s,%.2f,%d\n", curr->item_name, curr->price, curr->quantity);
        if (n < 0) {
            return EXIT_FAILURE;
        }
        curr = curr->next;
    }
    fclose(fp);
    return EXIT_SUCCESS;
}

// load the list from file filename
// the file should be in the following format:
// item_name,price,quantity\n 
//   (one item per line, separated by commas, and newline at the end)
// the loaded values are added to the end of the list
int list_load(node **head, char *filename)
{
    if (head == NULL || filename == NULL) {
        return EXIT_FAILURE;
    }
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        return EXIT_FAILURE;
    }

    int start = 1;
    // Get position of end of list
    node *curr = *head;
    while (curr) {
        curr = curr->next;
        start++;
    }
    int i = start;
    float price;
    int quantity;
    char *item_name = NULL;
    char *line = NULL;
    size_t linecap = 0;
    while (getline(&line, &linecap, fp) > 0) {
        item_name = realloc(item_name, linecap);
        int n = sscanf(line, "%[^,],%f,%d", item_name, &price, &quantity);
        if (n != 3 || 
            list_add_item_at_pos(head, item_name, price, quantity, i) == EXIT_FAILURE) {
            goto cleanup_failure;
        }
        i++;
    }

    free(item_name);
    free(line);
    fclose(fp);
    return EXIT_SUCCESS;

cleanup_failure:
    for (int j = i-1; j >= start; j--) {
        list_remove_item_at_pos(head, j);
    }
    free(item_name);
    free(line);
    fclose(fp);
    return EXIT_FAILURE;
}

struct node_sort_data {
    node *node;
    int pos;
};

// compare function for qsort (sort on item name)
int node_sort_compare(const void *a, const void *b)
{
    struct node_sort_data *data_a = (struct node_sort_data *)a;
    struct node_sort_data *data_b = (struct node_sort_data *)b;
    return strcmp(data_a->node->item_name, data_b->node->item_name);
}

// de-duplicate the list by combining items with the same name 
//    by adding their quantities
// The order of the returned list is undefined and may be in any order
int list_deduplicate(node **head) 
{
    if (head == NULL) {
        return EXIT_FAILURE;
    }
    struct node_sort_data *data = NULL;
    int len = 0;
    node *curr = *head;
    while (curr != NULL) {
        len++;
        curr = curr->next;
    }
    // Empty or singleton list is already deduplicated
    if (len == 0 || len == 1) {
        return EXIT_SUCCESS;
    }
    // Alloc the temp array for sorting
    data = malloc(sizeof(struct node_sort_data) * len);
    if (data == NULL) {
        return EXIT_FAILURE;
    }
    curr = *head;
    int i = 1;
    while (curr != NULL) {
        data[i-1].node = curr;
        data[i-1].pos = i;
        curr = curr->next;
        i++;
    }
    // Sort the array
    qsort(data, len, sizeof(struct node_sort_data), node_sort_compare);

    // Change the links to their sorted order
    *head = data[0].node;
    for (i = 0; i < len - 1; i++) {
        data[i].node->next = data[i+1].node;
    }
    data[len-1].node->next = NULL;
    free(data);

    // Merge the sorted array
    curr = *head;
    while (curr->next != NULL) {
        if (strcmp(curr->item_name, curr->next->item_name) == 0) {
            node *temp = curr->next;
            curr->quantity += curr->next->quantity;
            curr->next = curr->next->next;
            free(temp->item_name);
            free(temp);
        } else {
            curr = curr->next;
        }
    }
    return EXIT_SUCCESS;
}
